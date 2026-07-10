"""Real-hardware mission: crazyswarm2 cflib (radio) backend + our 4 nodes.

Only two things differ from sim.launch.py: the crazyswarm2 backend
(backend:=cflib instead of sim) and the perception node
(real_perception_node instead of sim_perception_node, which streams AI-deck
WiFi video + runs ArUco instead of checking a ground-truth file).
control_node and gcs_dashboard are unchanged -- same message contracts.

NOTE: `mocap` is left False, i.e. drones fly on their own onboard state
estimate over radio (no external motion-capture / lighthouse positioning).
Position will drift over time on real hardware without an external position
source -- that is a real limitation of this baseline, not something solved
here. Also NOTE: fill in each AI-deck's actual WiFi AP IP address in
mission_bringup/config/ai_deck_ips.yaml, and calibrate
cf_perception/config/camera_intrinsics.yaml before trusting marker detections.

IMPORTANT: `initial_position` is auto-computed as the first cell of each
drone's own assigned zone (same computation control_node itself runs -- see
_compute_homes below), but on real hardware YOU still have to physically
place each Crazyflie at that exact spot before launch -- unlike sim, nothing
here moves the physical drone there.
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

# Must match control_node's `dead_zone_margin` parameter default -- see
# sim.launch.py's _compute_homes for why both sides compute independently.
DEAD_ZONE_MARGIN = 0.15


def _enabled_drone_ids(crazyflies_cfg):
    """Same source as sim.launch.py -- see there for why."""
    return [
        drone_id for drone_id, robot_cfg in crazyflies_cfg.get('robots', {}).items()
        if robot_cfg.get('enabled', True)
    ]


def _compute_homes(mission_map, drone_ids):
    """Same computation as sim.launch.py -- see there for why."""
    boundary = [tuple(p) for p in mission_map['boundary']]
    dead_zones = [[tuple(p) for p in dz['points']] for dz in mission_map.get('dead_zones', [])]
    cells = build_cells(
        boundary, dead_zones, mission_map['coverage_line_spacing'], DEAD_ZONE_MARGIN)
    zone_cells = assign_cells_to_drones(cells, drone_ids)
    homes = {}
    for drone_id in drone_ids:
        waypoints = plan_coverage(zone_cells[drone_id])
        homes[drone_id] = waypoints[0] if waypoints else (0.0, 0.0)
    return homes


def _generate_crazyflies_yaml(crazyflies_cfg, homes, base_crazyflies_path):
    """Same injection as sim.launch.py -- see there for why."""
    for drone_id, (x, y) in homes.items():
        if drone_id in crazyflies_cfg.get('robots', {}):
            crazyflies_cfg['robots'][drone_id]['initial_position'] = [x, y, 0.0]

    fd, generated_path = tempfile.mkstemp(prefix='crazyflies_generated_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump(crazyflies_cfg, f)
    return generated_path


def _build(context, *args, **kwargs):
    bringup_share = get_package_share_directory('mission_bringup')
    perception_share = get_package_share_directory('cf_perception')
    mission_map_path = os.path.join(bringup_share, 'config', 'mission_map.yaml')
    base_crazyflies_path = os.path.join(bringup_share, 'config', 'crazyflies.yaml')
    ai_deck_ips_path = os.path.join(bringup_share, 'config', 'ai_deck_ips.yaml')
    camera_intrinsics_path = os.path.join(
        perception_share, 'config', 'camera_intrinsics.yaml')

    with open(base_crazyflies_path, 'r') as f:
        crazyflies_cfg = yaml.safe_load(f)
    drone_ids = _enabled_drone_ids(crazyflies_cfg)

    with open(mission_map_path, 'r') as f:
        mission_map = yaml.safe_load(f)
    homes = _compute_homes(mission_map, drone_ids)
    generated_crazyflies_path = _generate_crazyflies_yaml(
        crazyflies_cfg, homes, base_crazyflies_path)

    with open(ai_deck_ips_path, 'r') as f:
        ai_deck_ips = yaml.safe_load(f)
    missing = [d for d in drone_ids if d not in ai_deck_ips]
    if missing:
        raise RuntimeError(
            f'ai_deck_ips.yaml is missing an entry for: {missing} -- add each '
            "drone's AI-deck WiFi AP IP there before launching on real hardware")
    wifi_ips = [ai_deck_ips[d] for d in drone_ids]

    crazyswarm2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('crazyflie'), 'launch', 'launch.py')),
        launch_arguments={
            'backend': 'cflib',
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
        executable='real_perception_node',
        name='real_perception_node',
        output='screen',
        parameters=[{
            'drone_ids': drone_ids,
            'wifi_ips': wifi_ips,
            'wifi_port': 5000,
            'marker_size': 0.14,
            'camera_intrinsics_path': camera_intrinsics_path,
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
            'port': 5000,
        }],
    )

    return [crazyswarm2_launch, control_node, perception_node, gcs_node]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=_build)])
