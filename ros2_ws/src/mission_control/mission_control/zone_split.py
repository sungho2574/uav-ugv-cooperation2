"""Naive cellular decomposition: grid cells only, no polygon geometry.

Lays a coverage_line_spacing x coverage_line_spacing grid over the mission
boundary's bounding box, keeps only cells whose center is inside the
boundary and outside every dead zone, then hands roughly-equal, contiguous
left-to-right column bands of those cells to each drone -- as close to a
plain 3-way split as cells allow. Bands are matched to drones by plain list
order (the Nth drone gets the Nth region, left to right) -- there's no
"home position" to match against, since each drone's actual home/spawn point
is *derived from* its assigned band (see coverage_plan.plan_coverage), not
the other way around. Swappable baseline; a smarter decomposition algorithm
will replace this module later.
"""
import math

from shapely.geometry import Point, Polygon


def build_cells(boundary_points, dead_zone_point_lists, cell_size, dead_zone_margin=0.0):
    """Return every valid cell as a dict: {col, row, x, y} (x, y = cell center).

    A cell counts as valid if its center lands inside the boundary and
    outside every dead zone -- no attempt to handle cells that straddle an
    edge more precisely than that. `dead_zone_margin` buffers each dead zone
    outward first (mitre/sharp-cornered join, not rounded) so cells right at
    the edge of an obstacle still get excluded with some real clearance.
    """
    boundary = Polygon(boundary_points)
    dead_zones = [Polygon(pts) for pts in dead_zone_point_lists]
    if dead_zone_margin > 0:
        dead_zones = [dz.buffer(dead_zone_margin, join_style=2) for dz in dead_zones]
    minx, miny, maxx, maxy = boundary.bounds

    cells = []
    row = 0
    y = miny + cell_size / 2
    while y < maxy:
        col = 0
        x = minx + cell_size / 2
        while x < maxx:
            pt = Point(x, y)
            if boundary.contains(pt) and not any(dz.contains(pt) for dz in dead_zones):
                cells.append({'col': col, 'row': row, 'x': x, 'y': y})
            x += cell_size
            col += 1
        y += cell_size
        row += 1
    return cells


def assign_cells_to_drones(cells, drone_ids):
    """drone_ids: ordered list of drone id strings, e.g. ['cf1', 'cf2', 'cf3'].

    Splits the set of occupied columns into len(drone_ids) contiguous bands
    of (nearly) equal column count, left to right, and matches band i to
    drone_ids[i] -- the 1st drone in the list gets the leftmost region, the
    2nd gets the next one over, and so on. Returns dict drone_id -> list of
    cell dicts.
    """
    num_zones = len(drone_ids)
    cols = sorted({c['col'] for c in cells})
    band_size = math.ceil(len(cols) / num_zones) if cols else 1
    col_band = {col: min(i // band_size, num_zones - 1) for i, col in enumerate(cols)}

    bands = [[] for _ in range(num_zones)]
    for cell in cells:
        bands[col_band[cell['col']]].append(cell)

    return {drone_id: band for drone_id, band in zip(drone_ids, bands)}
