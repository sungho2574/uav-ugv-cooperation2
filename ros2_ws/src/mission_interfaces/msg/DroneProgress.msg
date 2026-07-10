string drone_id
int32 waypoint_index
int32 total_waypoints
# Every waypoint index actually confirmed visited (live position came within
# arrival_radius of it), independent of path order -- unlike waypoint_index
# (a highwater mark used for the "swept so far" tube), this can have gaps if
# the drone cut a corner and never got close enough to one cell in the
# middle of the path. GCS cell-darkening uses this set directly so a missed
# cell doesn't block every later cell from being marked visited.
int32[] visited_indices
