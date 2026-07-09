"""Naive ㄹ-shape (boustrophedon) coverage over an assigned cell list.

Visits every cell's center, row by row, alternating direction each row --
classic lawnmower/zigzag. No polygon geometry and no hole-avoidance detours:
if a cell is missing from the middle of a row (e.g. a dead zone cuts through
it), it's simply skipped and the row continues straight to the next present
cell. Keeping dead zones near a map edge/corner in the mission map is what
keeps that from creating an awkward mid-row jump -- not any code-level
handling here.

The path always starts at `start_xy` (the drone's home/spawn point): takeoff
only rises straight up, so the drone is already there right after takeoff,
and the sweep can begin immediately with no extra commute out to the first
cell.
"""


def plan_coverage(cells, start_xy):
    """cells: list of {col, row, x, y} dicts (see zone_split.build_cells).
    Returns ordered [(x, y), ...] waypoints, starting with start_xy itself."""
    waypoints = [start_xy]
    if not cells:
        return waypoints
    rows = sorted({c['row'] for c in cells})
    row_y = {c['row']: c['y'] for c in cells}
    # Start from whichever end of the zone (bottom-most or top-most row) is
    # closer to home -- otherwise a home near the top of its zone would sweep
    # bottom-up regardless and start with one huge diagonal leg clear across
    # the zone before the actual row-by-row coverage even begins.
    if abs(row_y[rows[0]] - start_xy[1]) > abs(row_y[rows[-1]] - start_xy[1]):
        rows = rows[::-1]
    left_to_right = True
    for row in rows:
        row_cells = sorted(
            (c for c in cells if c['row'] == row),
            key=lambda c: c['col'], reverse=not left_to_right)
        waypoints.extend((c['x'], c['y']) for c in row_cells)
        left_to_right = not left_to_right
    return waypoints
