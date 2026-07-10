"""One-call SCoPP mission planning facade: boundary -> per-drone zone + path.

Single entry point wrapping the three stages so control_node and both launch
files (sim/real _compute_homes) run byte-for-byte the same computation instead
of each re-orchestrating grid + allocation + path planning independently:

    grid  = build_grid(boundary, dead_zones, cell_width, margin)   # grid.py
    seeds = seed_positions(grid, len(drone_ids))                   # grid.py
    alloc = allocate(grid, seeds, bias)                            # area_allocation.py
    path  = plan_node_path(grid, owned, seed, profile)            # path_planning.py

Returns an ordered dict drone_id -> MissionPlan, where drone i is matched to
seed i / allocation node i (drone_ids order is authoritative -- it comes from
crazyflies.yaml, the single source of truth for which drones fly).
"""
from mission_control.area_allocation import AUCTION_BIAS, allocate
from mission_control.grid import build_grid, seed_positions
from mission_control.path_planning import plan_node_path


class MissionPlan:
    def __init__(self, cells, waypoints, home):
        self.cells = cells          # [Cell, ...] this drone owns (row-major)
        self.waypoints = waypoints  # [(x, y), ...] to sweep; [0] == home
        self.home = home            # (x, y)


def plan_mission(boundary_points, dead_zone_point_lists, cell_width, drone_ids,
                 dead_zone_margin=0.0, bias=AUCTION_BIAS, profile='paper_nn'):
    """Plan zones + coverage paths for every drone. Returns {drone_id: MissionPlan}.

    boundary_points / dead_zone_point_lists: [(x, y), ...] polygon rings.
    drone_ids: ordered ids from crazyflies.yaml; seed i -> drone i.
    """
    grid = build_grid(boundary_points, dead_zone_point_lists, cell_width, dead_zone_margin)
    seeds = seed_positions(grid, len(drone_ids))
    alloc = allocate(grid, seeds, bias)

    plans = {}
    for i, drone_id in enumerate(drone_ids):
        owned = alloc.cells_by_node.get(i, [])
        seed = seeds[i] if i < len(seeds) else (0.0, 0.0)
        node_path = plan_node_path(grid, owned, seed, profile=profile)
        home = node_path.home if node_path.coverage_waypoints else (seed[0], seed[1])
        plans[drone_id] = MissionPlan(
            cells=owned, waypoints=node_path.coverage_waypoints, home=home)
    return plans
