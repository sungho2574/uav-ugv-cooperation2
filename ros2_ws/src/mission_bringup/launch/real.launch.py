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
here. Also NOTE: wifi_ips below are placeholders -- set them to each AI-deck's
actual WiFi AP IP address, and calibrate cf_perception/config/camera_intrinsics.yaml
before trusting marker detections.

IMPORTANT: `initial_position` is auto-generated from mission_map.yaml's
home_position for each drone (see _generate_crazyflies_yaml below), but on
real hardware YOU still have to physically place each Crazyflie at that exact
spot before launch -- unlike sim, nothing here moves the physical drone there.
"""
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def _generate_crazyflies_yaml(mission_map, base_crazyflies_path):
    """Same injection as sim.launch.py -- see there for why."""
    with open(base_crazyflies_path, 'r') as f:
        crazyflies_cfg = yaml.safe_load(f)
    for d in mission_map['drones']:
        if d['id'] in crazyflies_cfg.get('robots', {}):
            crazyflies_cfg['robots'][d['id']]['initial_position'] = list(d['home_position'])

    fd, generated_path = tempfile.mkstemp(prefix='crazyflies_generated_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump(crazyflies_cfg, f)
    return generated_path


def _build(context, *args, **kwargs):
    bringup_share = get_package_share_directory('mission_bringup')
    perception_share = get_package_share_directory('cf_perception')
    mission_map_path = os.path.join(bringup_share, 'config', 'mission_map.yaml')
    base_crazyflies_path = os.path.join(bringup_share, 'config', 'crazyflies.yaml')
    camera_intrinsics_path = os.path.join(
        perception_share, 'config', 'camera_intrinsics.yaml')

    with open(mission_map_path, 'r') as f:
        mission_map = yaml.safe_load(f)
    generated_crazyflies_path = _generate_crazyflies_yaml(mission_map, base_crazyflies_path)
    drone_ids = [d['id'] for d in mission_map['drones']]
    # TODO: replace with each AI-deck's actual WiFi AP IP address.
    wifi_ips = ['192.168.4.1', '192.168.4.2', '192.168.4.3']

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
