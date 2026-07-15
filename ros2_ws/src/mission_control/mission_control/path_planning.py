"""SCoPP path planning over a node's assigned cells.

Port of docs/portable_area_allocation_and_path_planning.md part 2. Given the
cells one drone owns, decide a visiting order then stitch it into an actually-
flyable trajectory that stays on the 4-neighbour grid (so it naturally routes
around dead zones / holes rather than cutting straight across them).

Two visiting-order profiles:
  - 'paper_nn'   : the paper's original greedy nearest-neighbour (KD-tree),
                   cheap and deterministic. Default.
  - 'metric_tsp' : optimize order using grid shortest-path distance as the TSP
                   metric -- Held-Karp exact for <=20 targets, cheapest-
                   insertion + 2-opt above that. Shorter routes, but building
                   the distance matrix runs an A* between every pair of owned
                   cells, so it costs a few seconds for large zones.

Both share the same 4-neighbour A* stitcher (adjacent_path) and the same
distance/trajectory accounting. Every tie-break resolves to the smaller
stable_index, so results are reproducible for a fixed allocation.

`adjacent_path` searches only within the node's OWNED cells, so each drone's
whole route stays inside its own zone. If the owned region is split in two by
an obstacle, no path exists between the pieces -- rather than fail the whole
mission, the stitcher falls back to a direct jump to that target (see
plan_node_path) so coverage still proceeds; that jump is a straight line the
drone flies over, which is acceptable degradation for this baseline.
"""
import heapq
import math

NEIGHBOUR_STEPS = ((-1, 0), (0, -1), (0, 1), (1, 0))  # up, left, right, down -- fixed order
TSP_EXACT_MAX_TARGETS = 20   # doc's Held-Karp practical ceiling


class NoAdjacentPathError(Exception):
    """No 4-neighbour path exists between two cells within the valid set."""


class _NodeContext:
    """The subset of the grid one drone plans over: its owned cells only."""
    def __init__(self, grid, owned_cells):
        self.grid = grid
        self.cells = list(owned_cells)
        self.cell_by_id = {c.id: c for c in self.cells}
        self.valid_keys = {(c.row, c.col) for c in self.cells}
        self.id_by_key = {(c.row, c.col): c.id for c in self.cells}


def _sqdist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def adjacent_path(ctx, start_id, goal_id):
    """4-direction, unit-cost, Manhattan-heuristic A* between two owned cells.

    Returns a list of cell ids from start to goal inclusive. The fixed
    neighbour order + an insertion sequence counter in the priority queue make
    the chosen path deterministic among equal-cost alternatives. Raises
    NoAdjacentPathError if the two cells aren't connected within ctx."""
    if start_id == goal_id:
        return [start_id]
    start = ctx.cell_by_id[start_id]
    goal = ctx.cell_by_id[goal_id]
    start_key = (start.row, start.col)
    goal_key = (goal.row, goal.col)

    frontier = [(0, 0, 0, start_key)]     # (f, g, insertion_seq, key)
    came_from = {start_key: None}
    cost = {start_key: 0}
    seq = 0

    while frontier:
        _, g, _, current = heapq.heappop(frontier)
        if current == goal_key:
            path = []
            k = current
            while k is not None:
                path.append(ctx.id_by_key[k])
                k = came_from[k]
            path.reverse()
            return path
        if g != cost[current]:
            continue                       # stale queue entry
        cr, cc = current
        for dr, dc in NEIGHBOUR_STEPS:
            neighbour = (cr + dr, cc + dc)
            if neighbour not in ctx.valid_keys:
                continue
            new_g = g + 1
            if new_g < cost.get(neighbour, math.inf):
                cost[neighbour] = new_g
                came_from[neighbour] = current
                seq += 1
                h = abs(neighbour[0] - goal_key[0]) + abs(neighbour[1] - goal_key[1])
                heapq.heappush(frontier, (new_g + h, new_g, seq, neighbour))

    raise NoAdjacentPathError(f'{start_id} -> {goal_id}')


def start_cell(ctx, node_position):
    """Nearest owned cell center to the drone's real start position (tie by
    smaller stable_index). This cell's center doubles as the drone's home."""
    return min(
        ctx.cells,
        key=lambda c: (_sqdist(c.center, node_position), c.stable_index))


# --------------------------------------------------------------------------- #
# Profile A: paper_nn -- greedy nearest neighbour via a deterministic KD-tree
# --------------------------------------------------------------------------- #
class _KDNode:
    __slots__ = ('cell', 'axis', 'left', 'right')

    def __init__(self, cell, axis, left, right):
        self.cell = cell
        self.axis = axis
        self.left = left
        self.right = right


def _build_kdtree(cells, depth=0):
    if not cells:
        return None
    axis = depth % 2
    other = 1 - axis
    ordered = sorted(cells, key=lambda c: (c.center[axis], c.center[other], c.stable_index))
    mid = len(ordered) // 2
    return _KDNode(
        cell=ordered[mid],
        axis=axis,
        left=_build_kdtree(ordered[:mid], depth + 1),
        right=_build_kdtree(ordered[mid + 1:], depth + 1),
    )


def _kd_nearest(root, target):
    best_cell = root.cell
    best_key = (_sqdist(root.cell.center, target), root.cell.stable_index)

    def visit(node):
        nonlocal best_cell, best_key
        if node is None:
            return
        key = (_sqdist(node.cell.center, target), node.cell.stable_index)
        if key < best_key:
            best_cell, best_key = node.cell, key
        delta = target[node.axis] - node.cell.center[node.axis]
        near, far = (node.left, node.right) if delta <= 0 else (node.right, node.left)
        visit(near)
        if delta * delta <= best_key[0]:
            visit(far)

    visit(root)
    return best_cell


def _ordered_nearest_neighbour(start_position, cells):
    remaining = list(cells)
    current = start_position
    route = []
    while remaining:
        tree = _build_kdtree(remaining)
        chosen = _kd_nearest(tree, current)
        route.append(chosen)
        current = chosen.center
        remaining = [c for c in remaining if c.id != chosen.id]
    return route


# --------------------------------------------------------------------------- #
# Profile B: metric_tsp -- grid-shortest-path metric, Held-Karp / insertion+2opt
# --------------------------------------------------------------------------- #
def _build_metric(ctx, depot_id, target_cells):
    """Distance matrix over [depot] + targets using grid A* hop counts * width.
    Unreachable pairs get math.inf (the order optimizers simply avoid them)."""
    ids = [depot_id] + [c.id for c in target_cells]
    n = len(ids)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            try:
                hops = len(adjacent_path(ctx, ids[i], ids[j])) - 1
                d = hops * ctx.grid.cell_width
            except NoAdjacentPathError:
                d = math.inf
            D[i][j] = D[j][i] = d
    return D


def _held_karp_cycle(D):
    n = len(D) - 1
    states = {}
    for t in range(1, n + 1):
        states[(1 << (t - 1), t)] = (D[0][t], 0)

    for size in range(2, n + 1):
        for mask in range(1 << n):
            if bin(mask).count('1') != size:
                continue
            for last in range(1, n + 1):
                bit = 1 << (last - 1)
                if not (mask & bit):
                    continue
                prev_mask = mask ^ bit
                best = None
                for prev in range(1, n + 1):
                    if not (prev_mask & (1 << (prev - 1))):
                        continue
                    if (prev_mask, prev) not in states:
                        continue
                    cost = states[(prev_mask, prev)][0] + D[prev][last]
                    if best is None or cost < best[0]:
                        best = (cost, prev)
                if best is not None:
                    states[(mask, last)] = best

    full = (1 << n) - 1
    best_last, best_total = None, None
    for last in range(1, n + 1):
        if (full, last) not in states:
            continue
        total = states[(full, last)][0] + D[last][0]
        if best_total is None or total < best_total:
            best_last, best_total = last, total

    order = []
    mask, last = full, best_last
    while last != 0:
        order.append(last)
        _, prev = states[(mask, last)]
        mask ^= (1 << (last - 1))
        last = prev
    order.reverse()
    return order


def _insertion_two_opt_cycle(D):
    n = len(D) - 1
    remaining = set(range(1, n + 1))
    first = min(remaining, key=lambda t: (D[0][t], t))
    order = [first]
    remaining.discard(first)

    while remaining:
        best = None   # (increase, item, position)
        for item in remaining:
            for position in range(len(order) + 1):
                prev = 0 if position == 0 else order[position - 1]
                nxt = 0 if position == len(order) else order[position]
                increase = D[prev][item] + D[item][nxt] - D[prev][nxt]
                cand = (increase, item, position)
                if best is None or cand < best:
                    best = cand
        _, item, position = best
        order.insert(position, item)
        remaining.discard(item)

    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                before = 0 if i == 0 else order[i - 1]
                after = 0 if j == len(order) - 1 else order[j + 1]
                delta = (D[before][order[j]] + D[order[i]][after]
                         - D[before][order[i]] - D[order[j]][after])
                if delta < -1e-9:
                    order[i:j + 1] = reversed(order[i:j + 1])
                    improved = True
                    break
            if improved:
                break
    return order


def _metric_tsp_order(ctx, depot_id, target_cells):
    D = _build_metric(ctx, depot_id, target_cells)
    n = len(target_cells)
    if n == 0:
        return []
    order = (_held_karp_cycle(D) if n <= TSP_EXACT_MAX_TARGETS
             else _insertion_two_opt_cycle(D))
    # order holds metric indices 1..n; map back to target cells
    return [target_cells[i - 1] for i in order]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
class NodePath:
    def __init__(self, coverage_waypoints, home, distance):
        self.coverage_waypoints = coverage_waypoints  # [(x, y), ...], [0] == home
        self.home = home                              # (x, y)
        self.distance = distance                      # meters, incl. return-to-home


def _stitch(ctx, depot_id, ordered_cells):
    """Turn a visiting order into grid-adjacent cell ids from the depot through
    every target. Missing links (disconnected owned region) degrade to a direct
    jump rather than aborting -- see module docstring."""
    motion_ids = [depot_id]
    current = depot_id
    for target in ordered_cells:
        try:
            segment = adjacent_path(ctx, current, target.id)
            motion_ids.extend(segment[1:])
        except NoAdjacentPathError:
            motion_ids.append(target.id)   # jump straight to it
        current = target.id
    return motion_ids


def plan_node_path(grid, owned_cells, node_position, profile='paper_nn'):
    """Plan one drone's coverage path over its owned cells.

    Returns a NodePath whose coverage_waypoints are the cell centers to sweep,
    starting at home (nearest owned cell to node_position). The return-to-home
    leg is NOT included in coverage_waypoints -- control_node flies that as its
    own RETURN_HOME phase -- but it IS counted into `distance`.
    """
    if not owned_cells:
        home = (node_position[0], node_position[1])
        return NodePath([], home, 0.0)

    ctx = _NodeContext(grid, owned_cells)
    depot = start_cell(ctx, node_position)

    if profile == 'metric_tsp':
        targets = sorted(owned_cells, key=lambda c: c.stable_index)
        ordered = _metric_tsp_order(ctx, depot.id, targets)
    else:  # 'paper_nn' (default / fallback)
        ordered = _ordered_nearest_neighbour(node_position, owned_cells)

    motion_ids = _stitch(ctx, depot.id, ordered)
    waypoints = [ctx.cell_by_id[cid].center for cid in motion_ids]

    # distance: real start -> first cell + along swept cells + last cell -> home
    home = depot.center
    distance = math.dist(node_position, waypoints[0]) if waypoints else 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        distance += math.dist(a, b)
    if waypoints:
        distance += math.dist(waypoints[-1], home)   # return leg (control_node flies it)

    return NodePath(waypoints, home, distance)
