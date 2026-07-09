"""Naive ㄹ-shape (boustrophedon) coverage over an assigned cell list.

Visits every cell's center, row by row (bottom to top), alternating direction
each row -- classic lawnmower/zigzag. No polygon geometry and no hole-
avoidance detours: if a cell is missing from the middle of a row (e.g. a dead
zone cuts through it), it's simply skipped and the row continues straight to
the next present cell. Keeping dead zones near a map edge/corner in the
mission map is what keeps that from creating an awkward mid-row jump -- not
any code-level handling here.

There's no separate "home position" fed in: the very first waypoint this
produces (the bottom-most row's first cell) *is* the drone's home/spawn
point -- see control_node.py, which takes waypoints[0] as home after
planning, and mission_bringup's launch files, which run this same function
to inject that same point as each drone's spawn position before crazyswarm2
even starts. Takeoff only rises straight up, so wherever the drone spawns is
wherever the sweep needs to begin.
"""


def plan_coverage(cells):
    """cells: list of {col, row, x, y} dicts (see zone_split.build_cells).
    Returns ordered [(x, y), ...] waypoints; waypoints[0] doubles as this
    zone's home/spawn point."""
    if not cells:
        return []
    rows = sorted({c['row'] for c in cells})
    waypoints = []
    left_to_right = True
    for row in rows:
        row_cells = sorted(
            (c for c in cells if c['row'] == row),
            key=lambda c: c['col'], reverse=not left_to_right)
        waypoints.extend((c['x'], c['y']) for c in row_cells)
        left_to_right = not left_to_right
    return waypoints
