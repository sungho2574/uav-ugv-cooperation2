"""Naive coverage path baseline: grid-cell boustrophedon (lawnmower) sweep.

The mission is to visit every `line_spacing` x `line_spacing` cell inside a
zone -- a waypoint is a *cell center*, not a point on the zone's boundary.
Cells are laid out on a grid anchored to the zone's own bounding-box origin,
swept row by row (alternating direction each row, lawnmower-style). Rows and
sub-polygons are visited in simple nearest-next order starting from the
drone's home position -- no attempt at optimizing total travel distance.
Swappable baseline; a smarter planner will replace this module later.

Any dead-zone hole is cut out of the zone *once*, up front (same vertical-strip
technique zone_split.py uses to split zones among drones), turning a
polygon-with-a-hole into a handful of simple, hole-free pieces before sweeping.
An earlier version instead tried to route around a hole separately on every
single scan row that crossed it -- for a hole a few rows tall that meant every
one of those rows detouring all the way to the zone's far top/bottom edge and
back, which is what produced the "climbs way up, drops back down" flight
pattern. Splitting once avoids that entirely: each piece gets its own plain
lawnmower sweep with no per-row special-casing at all.
"""
import math

from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box


def _split_out_holes(poly):
    """Return a list of hole-free polygons whose union is `poly`.

    Slices `poly` into vertical strips at the x-extent of each interior hole.
    Any strip that doesn't overlap a hole in y comes back as-is (hole-free);
    a strip that spans a hole's full width naturally comes back as two
    separate polygons (below/above the hole) once Shapely intersects it --
    no special-casing needed here.
    """
    if not poly.interiors:
        return [poly] if not poly.is_empty and poly.area > 1e-9 else []

    minx, miny, maxx, maxy = poly.bounds
    xs = {minx, maxx}
    for ring in poly.interiors:
        ring_minx, _, ring_maxx, _ = Polygon(ring).bounds
        xs.add(ring_minx)
        xs.add(ring_maxx)
    xs = sorted(xs)

    pieces = []
    for x0, x1 in zip(xs, xs[1:]):
        if x1 - x0 < 1e-9:
            continue
        strip = poly.intersection(box(x0, miny - 1.0, x1, maxy + 1.0))
        if strip.is_empty:
            continue
        if strip.geom_type == 'Polygon':
            candidates = [strip]
        elif strip.geom_type in ('MultiPolygon', 'GeometryCollection'):
            # GeometryCollection can show up when the strip's edge exactly
            # coincides with the hole's edge -- Shapely then also reports the
            # degenerate shared edge as a zero-area LineString/Point alongside
            # the real polygon pieces; just ignore anything non-polygonal.
            candidates = [g for g in strip.geoms if g.geom_type == 'Polygon']
        else:
            candidates = []
        pieces.extend(p for p in candidates if not p.is_empty and p.area > 1e-9)
    return pieces


def _ordered_subpolygons(zone_geom, start_xy):
    if isinstance(zone_geom, MultiPolygon):
        candidates = [p for p in zone_geom.geoms if not p.is_empty and p.area > 1e-9]
    elif zone_geom.is_empty or zone_geom.area <= 1e-9:
        candidates = []
    else:
        candidates = [zone_geom]

    polys = []
    for p in candidates:
        polys.extend(_split_out_holes(p))

    ordered = []
    remaining = list(polys)
    cur = Point(start_xy)
    while remaining:
        remaining.sort(key=lambda p: p.centroid.distance(cur))
        nxt = remaining.pop(0)
        ordered.append(nxt)
        cur = nxt.centroid
    return ordered


def _sweep_cells(poly, cell_size, grid_origin):
    """Return an ordered list of grid-cell *centers* covering `poly`.

    Cells are `cell_size` x `cell_size`, aligned to `grid_origin` (shared
    across all pieces of a zone so neighboring pieces' cells line up rather
    than each piece inventing its own grid from its own bounding box). A row
    is only ever a single contiguous run of cells because `poly` is hole-free
    (see _split_out_holes) -- no mid-row gaps to worry about.
    """
    ox, oy = grid_origin
    minx, miny, maxx, maxy = poly.bounds
    first_row = math.floor((miny - oy) / cell_size - 0.5)
    last_row = math.ceil((maxy - oy) / cell_size - 0.5)

    waypoints = []
    left_to_right = True
    for row in range(first_row, last_row + 1):
        y = oy + (row + 0.5) * cell_size
        if y < miny - cell_size or y > maxy + cell_size:
            continue
        line = LineString([(minx - 1.0, y), (maxx + 1.0, y)])
        inter = poly.intersection(line)
        if inter.is_empty or inter.geom_type != 'LineString':
            left_to_right = not left_to_right
            continue
        seg_minx, seg_maxx = inter.bounds[0], inter.bounds[2]
        first_col = math.ceil((seg_minx - ox) / cell_size - 0.5)
        last_col = math.floor((seg_maxx - ox) / cell_size - 0.5)
        cols = range(first_col, last_col + 1)
        if not left_to_right:
            cols = reversed(list(cols))
        for col in cols:
            x = ox + (col + 0.5) * cell_size
            waypoints.append((x, y))
        left_to_right = not left_to_right
    return waypoints


def _safe_transit(zone_geom, p1, p2):
    """Return extra waypoints (possibly none) to get from p1 to p2 without
    crossing a hole. Nearest-piece ordering can still place two pieces that
    flank the *same* hole back to back (e.g. "above the hole" right before
    "below the hole") where a straight line between them cuts straight
    through it, even though each piece's own sweep never does.

    Tries an L-shaped detour via each of the zone's four outer bounding-box
    edges (always hole-free, since holes are strictly interior) and picks
    the shortest one that's actually verified clear; only used when the
    direct line isn't already safe.
    """
    zone_buffered = zone_geom.buffer(1e-6)
    if zone_buffered.contains(LineString([p1, p2])):
        return []
    minx, miny, maxx, maxy = zone_geom.bounds
    candidates = []
    for y in (miny, maxy):
        candidates.append([p1, (p1[0], y), (p2[0], y), p2])
    for x in (minx, maxx):
        candidates.append([p1, (x, p1[1]), (x, p2[1]), p2])
    safe = [via for via in candidates if zone_buffered.contains(LineString(via))]
    pool = safe or candidates  # best-effort fallback if nothing verifies clean
    best = min(pool, key=lambda via: LineString(via).length)
    return best[1:3]


def plan_coverage(zone_geom, line_spacing, start_xy):
    """zone_geom: shapely Polygon/MultiPolygon. Returns ordered [(x, y), ...] cell-center
    waypoints, one per line_spacing x line_spacing cell inside the zone.

    The path always starts at `start_xy` itself (the drone's home/spawn point),
    not at whichever cell center happens to be nearest to it -- home is where
    the drone already is right after takeoff (straight up, same x/y), so the
    first leg of the sweep should be zero-distance rather than an extra
    unplanned commute out to the first cell.
    """
    if zone_geom.is_empty:
        return []
    grid_origin = (zone_geom.bounds[0], zone_geom.bounds[1])  # shared by every piece

    waypoints = [start_xy]
    last_point = start_xy
    for poly in _ordered_subpolygons(zone_geom, start_xy):
        piece_wps = _sweep_cells(poly, line_spacing, grid_origin)
        if not piece_wps:
            continue
        waypoints.extend(_safe_transit(zone_geom, last_point, piece_wps[0]))
        waypoints.extend(piece_wps)
        last_point = piece_wps[-1]
    return waypoints
