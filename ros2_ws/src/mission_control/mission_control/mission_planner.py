"""Single planning facade: pick the allocation/coverage algorithm from config.

This is the ONE entry point that control_node's PREPARE step and both launch
files' _compute_homes() all call, so they compute byte-for-byte the same
zones/paths/homes instead of each re-orchestrating the planner independently.
That shared computation matters because a drone's spawn point is fixed at
launch time (crazyswarm2 needs each drone's initial_position before the node
starts), and the launch files derive it from the *same* plan control_node
later flies -- if the two disagreed, drones would spawn away from their
planned home.

Two interchangeable algorithms, picked per mission_map.yaml's `planner` field:

  - "simple" (default): the naive baseline -- zone_split.build_cells +
    assign_cells_to_drones (vertical column bands) then coverage_plan.plan_coverage
    (boustrophedon sweep). Kept as the safe default; scopp is opt-in.
  - "scopp": SCoPP-style allocation (Lloyd clustering on cell perimeters +
    greedy auction, area_allocation.py) + path planning (KD-tree nearest
    neighbour or grid-metric TSP, path_planning.py) over a richer cell grid
    (grid.py). Its coverage path order is further tunable via the
    `coverage_profile` field ('paper_nn' default, or 'metric_tsp').

Both algorithms are normalized into the same ZonePlan shape so everything
downstream (control_node._publish_plan, trajectory building, the launch homes)
is algorithm-agnostic -- see ZonePlan.
"""
from mission_control.area_allocation import AUCTION_BIAS, allocate
from mission_control.coverage_plan import plan_coverage
from mission_control.grid import build_grid, seed_positions
from mission_control.path_planning import plan_node_path
from mission_control.zone_split import assign_cells_to_drones, build_cells


class ZonePlan:
    """One drone's planned zone + coverage path, in a form both algorithms
    (and every consumer) share. `cells` is deliberately the plain
    {col, row, x, y} dict shape the naive planner already used and that
    control_node._publish_plan reads directly (cell['x']/cell['y']) -- the
    scopp planner's richer Cell objects get converted down to it here so no
    consumer has to care which algorithm produced the plan."""

    def __init__(self, cells, waypoints, home):
        self.cells = cells          # [{'col', 'row', 'x', 'y'}, ...]
        self.waypoints = waypoints  # [(x, y), ...] to sweep; [0] == home
        self.home = home            # (x, y)


def plan_zones(mission_map, drone_ids, dead_zone_margin):
    """Plan zones + coverage paths for every drone. Returns {drone_id: ZonePlan}.

    Dispatches on mission_map['planner'] ('simple' default). boundary,
    dead_zones and cell width (== coverage_line_spacing) all come straight
    from mission_map, same as the callers used to read them themselves.
    """
    boundary = [tuple(p) for p in mission_map['boundary']]
    # `or []` (not just `.get(..., [])`): a bare `dead_zones:` key with nothing
    # under it (or `dead_zones: null`) parses to None in YAML, not a missing
    # key, so the [] default never kicks in and `for dz in None` crashes.
    dead_zones = [
        [tuple(p) for p in dz['points']] for dz in (mission_map.get('dead_zones') or [])
    ]
    cell_width = float(mission_map['coverage_line_spacing'])
    planner = mission_map.get('planner', 'simple')

    if planner == 'simple':
        return _plan_simple(boundary, dead_zones, cell_width, drone_ids, dead_zone_margin)
    elif planner == 'scopp':
        profile = mission_map.get('coverage_profile', 'paper_nn')
        return _plan_scopp(
            boundary, dead_zones, cell_width, drone_ids, dead_zone_margin, profile)
    else:
        raise RuntimeError(
            f"unknown planner '{planner}' in mission_map.yaml -- "
            "must be 'simple' or 'scopp'")


def _plan_simple(boundary, dead_zones, cell_width, drone_ids, dead_zone_margin):
    """Naive baseline: column-band split + boustrophedon sweep. Cells are
    already {col, row, x, y} dicts, so no conversion needed."""
    cells = build_cells(boundary, dead_zones, cell_width, dead_zone_margin)
    zone_cells = assign_cells_to_drones(cells, drone_ids)

    plans = {}
    for drone_id in drone_ids:
        owned = zone_cells[drone_id]
        waypoints = plan_coverage(owned)
        home = waypoints[0] if waypoints else (0.0, 0.0)
        plans[drone_id] = ZonePlan(cells=owned, waypoints=waypoints, home=home)
    return plans


def _cell_to_dict(cell):
    """SCoPP Cell (with .center/.row/.col) -> the {col, row, x, y} dict shape
    the naive planner and control_node._publish_plan expect."""
    return {'col': cell.col, 'row': cell.row, 'x': cell.center[0], 'y': cell.center[1]}


def _plan_scopp(boundary, dead_zones, cell_width, drone_ids, dead_zone_margin, profile):
    """SCoPP allocation + path planning. drone i is matched to seed i /
    allocation node i (drone_ids order is authoritative -- it comes from
    crazyflies.yaml, the single source of truth for which drones fly)."""
    grid = build_grid(boundary, dead_zones, cell_width, dead_zone_margin)
    seeds = seed_positions(grid, len(drone_ids))
    alloc = allocate(grid, seeds, AUCTION_BIAS)

    plans = {}
    for i, drone_id in enumerate(drone_ids):
        owned = alloc.cells_by_node.get(i, [])
        seed = seeds[i] if i < len(seeds) else (0.0, 0.0)
        node_path = plan_node_path(grid, owned, seed, profile=profile)
        home = node_path.home if node_path.coverage_waypoints else (seed[0], seed[1])
        plans[drone_id] = ZonePlan(
            cells=[_cell_to_dict(c) for c in owned],
            waypoints=node_path.coverage_waypoints,
            home=home)
    return plans
