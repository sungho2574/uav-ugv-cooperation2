"""SCoPP area allocation: Lloyd clustering on cell perimeters + greedy auction.

Port of docs/portable_area_allocation_and_path_planning.md part 1
(Collins et al., SCoPP, arXiv:2103.14709 Sec. III-C~III-D). Three stages:

  1. Lloyd (K-means) clustering of every cell's *perimeter* sample points,
     seeded from the drone start positions -- clustering the perimeters (not
     the centers) is what lets a single cell land in two clusters and be
     flagged a "conflict" cell.
  2. Cells whose perimeter samples all fell in one cluster are assigned to that
     drone immediately; cells split across >1 cluster become conflict cells.
  3. A greedy online auction resolves conflict cells one at a time in fixed
     row-major order, each going to the drone with the lowest bid
     (bid = cells_already_won + B * initial_distance_bias).

Fully deterministic: every tie -- nearest cluster, auction bid, anything --
resolves to the smaller cluster/drone index. Same input => same output.
"""
import math

AUCTION_BIAS = 0.5          # B in bid = N_r + B * d0(r); doc default 0.5
LLOYD_MAX_ITERATIONS = 10   # doc fixed value


def _sqdist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def lloyd_cluster(points, initial_centroids, tolerance, max_iterations=LLOYD_MAX_ITERATIONS):
    """Standard Lloyd iteration. Returns (centroids, labels).

    labels[i] is the cluster index of points[i]. Empty clusters keep their
    previous centroid (never collapse). Ties in the assignment step go to the
    smaller cluster index (deterministic)."""
    centroids = list(initial_centroids)
    n_clusters = len(centroids)
    labels = [0] * len(points)

    for _ in range(max_iterations):
        # assignment
        for pi, p in enumerate(points):
            best_i, best_d = 0, _sqdist(p, centroids[0])
            for i in range(1, n_clusters):
                d = _sqdist(p, centroids[i])
                if d < best_d:                 # strict < keeps the smaller index on ties
                    best_i, best_d = i, d
            labels[pi] = best_i

        # update
        sums = [[0.0, 0.0] for _ in range(n_clusters)]
        counts = [0] * n_clusters
        for pi, p in enumerate(points):
            c = labels[pi]
            sums[c][0] += p[0]
            sums[c][1] += p[1]
            counts[c] += 1
        new_centroids = []
        movement = 0.0
        for i in range(n_clusters):
            if counts[i] == 0:
                new_centroids.append(centroids[i])   # empty -> keep previous
            else:
                nc = (sums[i][0] / counts[i], sums[i][1] / counts[i])
                new_centroids.append(nc)
                movement = max(movement, math.dist(centroids[i], nc))
        centroids = new_centroids
        if movement <= tolerance:
            break

    # final relabel against the converged centroids
    for pi, p in enumerate(points):
        best_i, best_d = 0, _sqdist(p, centroids[0])
        for i in range(1, n_clusters):
            d = _sqdist(p, centroids[i])
            if d < best_d:
                best_i, best_d = i, d
        labels[pi] = best_i
    return centroids, labels


class AllocationResult:
    def __init__(self, owner_by_cell, cells_by_node, bias, d0_by_node):
        self.owner_by_cell = owner_by_cell    # cell_id -> node index
        self.cells_by_node = cells_by_node    # node index -> [Cell, ...] (row-major)
        self.bias = bias
        self.d0_by_node = d0_by_node          # per-node distance bias, for diagnostics


def allocate(grid, node_positions, bias=AUCTION_BIAS):
    """Assign every valid cell to exactly one node. See module docstring.

    node_positions: [(x, y), ...], index == node index == drone index.
    Returns an AllocationResult. Deterministic for a fixed grid + positions.
    """
    n_nodes = len(node_positions)
    if n_nodes == 0:
        return AllocationResult({}, {}, bias, [])
    if n_nodes == 1:
        only = list(grid.cells)
        return AllocationResult({c.id: 0 for c in only}, {0: only}, bias, [0.0])

    # --- stage 1: cluster all perimeter samples -------------------------------
    points = []
    point_cell_idx = []   # parallel: which cell each sample came from (index into grid.cells)
    for ci, cell in enumerate(grid.cells):
        for s in cell.perimeter_samples:
            points.append(s)
            point_cell_idx.append(ci)

    tolerance = grid.cell_width / 8.0
    _, labels = lloyd_cluster(points, node_positions, tolerance)

    # per-cell candidate cluster set (sorted, deterministic)
    per_cell_labels = [set() for _ in grid.cells]
    for pi, ci in enumerate(point_cell_idx):
        per_cell_labels[ci].add(labels[pi])
    cluster_candidates = [sorted(s) for s in per_cell_labels]

    # --- stage 2: split immediate vs conflict cells ---------------------------
    owner_by_cell = {}
    assigned_count = [0] * n_nodes
    initial_cells = [[] for _ in range(n_nodes)]     # single-owner cells per node
    for ci, cell in enumerate(grid.cells):
        cands = cluster_candidates[ci]
        if len(cands) == 1:
            owner = cands[0]
            owner_by_cell[cell.id] = owner
            assigned_count[owner] += 1
            initial_cells[owner].append(cell)

    conflict_cells = [
        grid.cells[ci] for ci in range(len(grid.cells))
        if len(cluster_candidates[ci]) > 1
    ]  # already in fixed row-major order (grid.cells order preserved)

    # --- stage 3: fixed per-node distance bias d0(r), computed once -----------
    d0 = [0.0] * n_nodes
    for r in range(n_nodes):
        pool = initial_cells[r]
        if not pool:
            # fallback: any cell that had r among its candidates (conflict incl.)
            pool = [grid.cells[ci] for ci in range(len(grid.cells))
                    if r in cluster_candidates[ci]]
        if not pool:
            d0[r] = 0.0
            continue
        nearest = min(math.dist(node_positions[r], c.center) for c in pool)
        d0[r] = round(nearest / grid.cell_width)

    # --- stage 3: greedy online auction over conflict cells -------------------
    cell_index = {cell.id: ci for ci, cell in enumerate(grid.cells)}
    for cell in conflict_cells:
        cands = cluster_candidates[cell_index[cell.id]]
        best_r, best_bid = None, None
        for r in cands:
            bid = assigned_count[r] + bias * d0[r]
            if best_bid is None or bid < best_bid:   # strict < keeps smaller index on ties
                best_r, best_bid = r, bid
        owner_by_cell[cell.id] = best_r
        assigned_count[best_r] += 1

    # --- collect per-node cell lists in fixed row-major order -----------------
    cells_by_node = {r: [] for r in range(n_nodes)}
    for cell in grid.cells:
        cells_by_node[owner_by_cell[cell.id]].append(cell)

    return AllocationResult(owner_by_cell, cells_by_node, bias, d0)
