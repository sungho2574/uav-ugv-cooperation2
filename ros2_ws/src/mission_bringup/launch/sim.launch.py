"""Simulated end-to-end mission: crazyswarm2 sim backend + our 4 nodes.

Brings up:
  - crazyswarm2's crazyflie_server (backend:=sim) for every `enabled` robot
    in mission_bringup/config/crazyflies.yaml (that file is the single
    source of truth for which drones fly -- see _enabled_drone_ids below)
  - mission_control/control_node          (central FSM)
  - cf_perception/sim_perception_node     (pose relay + ground-truth marker check)
  - gcs_dashboard/gcs_node                (Flask + Three.js dashboard, default port 5000)

Swapping to real hardware means swapping ONLY the crazyswarm2 backend and the
perception node (see real.launch.py) -- control_node, gcs_dashboard, and every
topic/message contract stay exactly the same.
"""
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from mission_control.coverage_plan import plan_coverage
from mission_control.zone_split import assign_cells_to_drones, build_cells

# Must match control_node's `dead_zone_margin` parameter default -- both sides
# run the exact same build_cells()/assign_cells_to_drones()/plan_coverage()
# computation independently (rather than passing data between the launch
# script and the node) so they always agree on where each drone's zone (and
# therefore its home/spawn point) actually is.
DEAD_ZONE_MARGIN = 0.15


def _enabled_drone_ids(crazyflies_cfg):
    """crazyswarm2 already requires crazyflies.yaml to exist (radio uri/type
    per robot), so it's the single source of truth for which drones fly --
    not mission_map.yaml. Order follows the file's robots: mapping order,
    which is also what assigns Nth-drone -> Nth-region in zone_split.py."""
    return [
        drone_id for drone_id, robot_cfg in crazyflies_cfg.get('robots', {}).items()
        if robot_cfg.get('enabled', True)
    ]


def _compute_homes(mission_map, drone_ids):
    """Each drone's home/spawn point is the first cell of its own assigned
    zone (see coverage_plan.py) -- e.g. the drone assigned the 3rd region
    spawns inside the 3rd region, not at some unrelated point that then
    needs a long straight commute to reach its own zone."""
    boundary = [tuple(p) for p in mission_map['boundary']]
    # `.get(..., [])` alone isn't enough: a bare `dead_zones:` key with no
    # entries under it (or `dead_zones: null`) parses to a real None value in
    # YAML, not a missing key, so the [] default never kicks in and `for dz
    # in None` crashes. `or []` catches both "key absent" and "key present
    # but empty/null".
    dead_zones = [[tuple(p) for p in dz['points']] for dz in (mission_map.get('dead_zones') or [])]
    cells = build_cells(
        boundary, dead_zones, mission_map['coverage_line_spacing'], DEAD_ZONE_MARGIN)
    zone_cells = assign_cells_to_drones(cells, drone_ids)
    homes = {}
    for drone_id in drone_ids:
        waypoints = plan_coverage(zone_cells[drone_id])
        homes[drone_id] = waypoints[0] if waypoints else (0.0, 0.0)
    return homes


def _generate_crazyflies_yaml(crazyflies_cfg, homes, base_crazyflies_path):
    """Overwrite each robot's initial_position with its computed home point,
    so the single source of truth for "where does this drone start" is the
    same zone/path computation control_node itself uses -- takeoff only
    rises straight up, so the spawn point and the path's start point must
    always match or the drone would need to "commute" sideways afterward.
    `enabled` is left exactly as loaded from crazyflies.yaml -- that file
    decides which drones fly, this function only fills in where.
    """
    for drone_id, (x, y) in homes.items():
        if drone_id in crazyflies_cfg.get('robots', {}):
            crazyflies_cfg['robots'][drone_id]['initial_position'] = [x, y, 0.0]

    fd, generated_path = tempfile.mkstemp(prefix='crazyflies_generated_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump(crazyflies_cfg, f)
    return generated_path


def _build(context, *args, **kwargs):
    bringup_share = get_package_share_directory('mission_bringup')
    mission_map_path = os.path.join(bringup_share, 'config', 'mission_map.yaml')
    true_markers_path = os.path.join(bringup_share, 'config', 'true_markers.yaml')
    base_crazyflies_path = os.path.join(bringup_share, 'config', 'crazyflies.yaml')

    with open(base_crazyflies_path, 'r') as f:
        crazyflies_cfg = yaml.safe_load(f)
    drone_ids = _enabled_drone_ids(crazyflies_cfg)

    with open(mission_map_path, 'r') as f:
        mission_map = yaml.safe_load(f)
    homes = _compute_homes(mission_map, drone_ids)
    generated_crazyflies_path = _generate_crazyflies_yaml(
        crazyflies_cfg, homes, base_crazyflies_path)

    crazyswarm2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('crazyflie'), 'launch', 'launch.py')),
        launch_arguments={
            'backend': 'sim',
            'crazyflies_yaml_file': generated_crazyflies_path,
            'mocap': 'False',
            'gui': 'False',
            'teleop': 'False',
            'rviz': 'False',
        }.items(),
    )

    control_node = Node(
        package='mission_control',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[{
            'mission_map_path': mission_map_path,
            'drone_ids': drone_ids,
            'start_immediately': False,
        }],
    )

    perception_node = Node(
        package='cf_perception',
        executable='sim_perception_node',
        name='sim_perception_node',
        output='screen',
        parameters=[{
            'drone_ids': drone_ids,
            'mission_map_path': mission_map_path,
            'true_markers_path': true_markers_path,
        }],
    )

    gcs_node = Node(
        package='gcs_dashboard',
        executable='gcs_node',
        name='gcs_node',
        output='screen',
        parameters=[{
            'drone_ids': drone_ids,
            'mission_map_path': mission_map_path,
            'true_markers_path': true_markers_path,
            'port': 5000,
        }],
    )

    return [crazyswarm2_launch, control_node, perception_node, gcs_node]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=_build)])
