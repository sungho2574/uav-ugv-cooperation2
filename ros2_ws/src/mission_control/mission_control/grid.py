"""Cell grid construction for SCoPP-style area allocation + path planning.

Grids the mission boundary into equal square cells (cell_width == the mission
map's coverage_line_spacing), keeping only cells whose center is inside the
boundary and outside every dead zone -- same "is this cell valid" test the old
zone_split.build_cells used. On top of that plain center test, this adds the
two extra pieces the SCoPP algorithms need (see
docs/portable_area_allocation_and_path_planning.md sec. 0):

  - each cell carries `perimeter_samples`, points sampled along the cell's four
    edges, which allocation clusters on (NOT the centers) so a cell straddling
    two clusters' boundary can be detected as a "conflict" cell.
  - every cell has a fixed `stable_index` (its position in the row-major
    `cells` list). ALL tie-breaks downstream resolve to the smaller stable
    index, so this ordering must never change for a given input.
"""
import math
from dataclasses import dataclass, field

from shapely.geometry import Point, Polygon

# Perimeter sampling density: each cell edge of length W is sampled at
# ceil(W / (W/8)) == 8 points. The doc's default `spacing = W / 8`.
PERIMETER_SAMPLES_PER_SIDE = 8


@dataclass
class Cell:
    id: str                       # deterministic "r{row}c{col}" id
    row: int
    col: int
    center: tuple                 # (x, y), meters, local Cartesian
    stable_index: int             # position in Grid.cells (row-major); tie-break key
    perimeter_samples: list = field(default_factory=list)  # [(x, y), ...]


@dataclass
class Grid:
    cells: list                   # [Cell, ...] in fixed row-major order
    cell_width: float
    id_by_key: dict               # (row, col) -> cell_id, for 4-neighbour lookup
    cell_by_id: dict              # cell_id -> Cell

    def neighbour_id(self, row, col):
        return self.id_by_key.get((row, col))


def _perimeter_samples(cx, cy, half):
    """Sample points along the four edges of a cell centered at (cx, cy) with
    half-width `half`. Points are taken at the mid of each of the
    PERIMETER_SAMPLES_PER_SIDE segments per side (so corners aren't double
    counted between adjacent sides); that's enough edge coverage for the
    conflict-cell test without depending on exact corner hits."""
    n = PERIMETER_SAMPLES_PER_SIDE
    lo_x, hi_x = cx - half, cx + half
    lo_y, hi_y = cy - half, cy + half
    pts = []
    for k in range(n):
        t = (k + 0.5) / n
        x = lo_x + t * (hi_x - lo_x)
        y = lo_y + t * (hi_y - lo_y)
        pts.append((x, lo_y))   # bottom edge
        pts.append((x, hi_y))   # top edge
        pts.append((lo_x, y))   # left edge
        pts.append((hi_x, y))   # right edge
    return pts


def build_grid(boundary_points, dead_zone_point_lists, cell_width, dead_zone_margin=0.0):
    """Return a Grid of every valid cell.

    A cell is valid if its center lands inside the boundary and outside every
    dead zone -- identical validity rule to the old zone_split.build_cells, so
    the exact same set of cells is produced, just wrapped in richer objects.
    `dead_zone_margin` buffers each dead zone outward first (mitre join) so
    cells hugging an obstacle edge still get excluded with real clearance.
    """
    boundary = Polygon(boundary_points)
    dead_zones = [Polygon(pts) for pts in dead_zone_point_lists]
    if dead_zone_margin > 0:
        dead_zones = [dz.buffer(dead_zone_margin, join_style=2) for dz in dead_zones]
    minx, miny, maxx, maxy = boundary.bounds
    half = cell_width / 2.0

    cells = []
    id_by_key = {}
    cell_by_id = {}
    row = 0
    y = miny + half
    while y < maxy:
        col = 0
        x = minx + half
        while x < maxx:
            pt = Point(x, y)
            if boundary.contains(pt) and not any(dz.contains(pt) for dz in dead_zones):
                cell = Cell(
                    id=f'r{row}c{col}',
                    row=row,
                    col=col,
                    center=(x, y),
                    stable_index=len(cells),
                    perimeter_samples=_perimeter_samples(x, y, half),
                )
                cells.append(cell)
                id_by_key[(row, col)] = cell.id
                cell_by_id[cell.id] = cell
            x += cell_width
            col += 1
        y += cell_width
        row += 1

    return Grid(cells=cells, cell_width=cell_width, id_by_key=id_by_key, cell_by_id=cell_by_id)


def seed_positions(grid, n):
    """Deterministic drone start-position seeds for allocation.

    The SCoPP algorithm needs a start position per drone to seed clustering,
    but this project derives each drone's home from its assigned zone instead
    of being given one up front. So we synthesize N seeds by splitting the
    grid's bounding box into N equal slices along its longer axis and snapping
    each slice's mid-point to the nearest (still-unused) valid cell center --
    a plain, reproducible spread that the Lloyd/auction stages then refine into
    balanced zones. Order matches drone_ids order: seed i -> drone i.

    Returns [(x, y), ...] of length n (fewer only if the grid has < n cells).
    """
    if not grid.cells or n <= 0:
        return []
    xs = [c.center[0] for c in grid.cells]
    ys = [c.center[1] for c in grid.cells]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    horizontal = (maxx - minx) >= (maxy - miny)

    seeds = []
    used = set()
    for i in range(n):
        f = (i + 0.5) / n
        if horizontal:
            target = (minx + f * (maxx - minx), (miny + maxy) / 2.0)
        else:
            target = ((minx + maxx) / 2.0, miny + f * (maxy - miny))
        candidates = [c for c in grid.cells if c.id not in used] or grid.cells
        best = min(
            candidates,
            key=lambda c: ((c.center[0] - target[0]) ** 2 + (c.center[1] - target[1]) ** 2,
                           c.stable_index))
        used.add(best.id)
        seeds.append(best.center)
    return seeds
