"""Naive coverage path baseline: boustrophedon (lawnmower) sweep.

For each sub-polygon of a (possibly multi-part) zone, sweeps horizontal scan
lines spaced `line_spacing` apart, alternating direction each line. Sub-polygons
and lines are visited in simple nearest-next order starting from the drone's
home position -- no attempt at optimizing total travel distance. Swappable
baseline; a smarter planner will replace this module later.
"""
from shapely.geometry import LineString, MultiPolygon, Point


def _ordered_subpolygons(zone_geom, start_xy):
    if isinstance(zone_geom, MultiPolygon):
        polys = [p for p in zone_geom.geoms if not p.is_empty and p.area > 1e-9]
    elif zone_geom.is_empty or zone_geom.area <= 1e-9:
        polys = []
    else:
        polys = [zone_geom]

    ordered = []
    remaining = list(polys)
    cur = Point(start_xy)
    while remaining:
        remaining.sort(key=lambda p: p.centroid.distance(cur))
        nxt = remaining.pop(0)
        ordered.append(nxt)
        cur = nxt.centroid
    return ordered


def _sweep_polygon(poly, line_spacing):
    """Return an ordered list of (x, y) waypoints covering `poly` with scan lines.

    A scan line can hit more than one disjoint segment when a dead-zone hole
    sits in the middle of it (left-of-hole / right-of-hole). Connecting those
    segments directly, end to end, would fly the straight-line "go_to" leg
    right over the hole. Instead we detour through whichever polygon edge
    (bottom/top of this sub-polygon's bounds) is closer -- dead-zone holes are
    strictly interior, so that edge is always clear -- before continuing the
    next segment. Adds extra travel but never overflies a dead zone.
    """
    minx, miny, maxx, maxy = poly.bounds
    waypoints = []
    left_to_right = True
    y = miny
    while y <= maxy + 1e-9:
        line = LineString([(minx - 1.0, y), (maxx + 1.0, y)])
        inter = poly.intersection(line)
        if not inter.is_empty:
            if inter.geom_type == 'LineString':
                segments = [inter]
            elif inter.geom_type == 'MultiLineString':
                segments = list(inter.geoms)
            else:
                segments = []
            segments.sort(key=lambda s: s.bounds[0])
            if not left_to_right:
                segments = segments[::-1]
            prev_x = None
            safe_y = miny if (y - miny) <= (maxy - y) else maxy
            for seg in segments:
                x0, x1 = seg.bounds[0], seg.bounds[2]
                if not left_to_right:
                    x0, x1 = x1, x0
                if prev_x is not None:
                    waypoints.append((prev_x, safe_y))
                    waypoints.append((x0, safe_y))
                waypoints.append((x0, y))
                waypoints.append((x1, y))
                prev_x = x1
        left_to_right = not left_to_right
        y += line_spacing
    return waypoints


def plan_coverage(zone_geom, line_spacing, start_xy):
    """zone_geom: shapely Polygon/MultiPolygon. Returns ordered [(x, y), ...] waypoints."""
    waypoints = []
    for poly in _ordered_subpolygons(zone_geom, start_xy):
        waypoints.extend(_sweep_polygon(poly, line_spacing))
    return waypoints
