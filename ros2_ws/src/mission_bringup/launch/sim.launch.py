"""Simulated end-to-end mission: crazyswarm2 sim backend + our 4 nodes.

Brings up:
  - crazyswarm2's crazyflie_server (backend:=sim) for cf1/cf2/cf3
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


def _generate_crazyflies_yaml(mission_map, base_crazyflies_path):
    """Overwrite each robot's initial_position with mission_map.yaml's
    home_position, so the single source of truth for "where does this drone
    start" is the mission map -- the drone's coverage path starts exactly at
    its own home_position (see coverage_plan.py), and takeoff only rises
    straight up, so the spawn point and the path's start point must always
    match or the drone would need to "commute" sideways after taking off.
    """
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
    mission_map_path = os.path.join(bringup_share, 'config', 'mission_map.yaml')
    true_markers_path = os.path.join(bringup_share, 'config', 'true_markers.yaml')
    base_crazyflies_path = os.path.join(bringup_share, 'config', 'crazyflies.yaml')

    with open(mission_map_path, 'r') as f:
        mission_map = yaml.safe_load(f)
    generated_crazyflies_path = _generate_crazyflies_yaml(mission_map, base_crazyflies_path)
    drone_ids = [d['id'] for d in mission_map['drones']]

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
