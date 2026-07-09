"""Simulated end-to-end mission: crazyswarm2 sim backend + our 4 nodes.

Brings up:
  - crazyswarm2's crazyflie_server (backend:=sim) for cf1/cf2/cf3, using our
    crazyflies.yaml (initial_position matches mission_map.yaml home_position)
  - mission_control/control_node          (central FSM)
  - cf_perception/sim_perception_node     (pose relay + ground-truth marker check)
  - gcs_dashboard/gcs_node                (Flask + Three.js dashboard, default port 5000)

Swapping to real hardware means swapping ONLY the crazyswarm2 backend and the
perception node (see real.launch.py) -- control_node, gcs_dashboard, and every
topic/message contract stay exactly the same.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory('mission_bringup')
    mission_map_path = os.path.join(bringup_share, 'config', 'mission_map.yaml')
    true_markers_path = os.path.join(bringup_share, 'config', 'true_markers.yaml')
    crazyflies_yaml_path = os.path.join(bringup_share, 'config', 'crazyflies.yaml')

    drone_ids = ['cf1', 'cf2', 'cf3']

    crazyswarm2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('crazyflie'), 'launch', 'launch.py')),
        launch_arguments={
            'backend': 'sim',
            'crazyflies_yaml_file': crazyflies_yaml_path,
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
            'port': 5000,
        }],
    )

    return LaunchDescription([
        crazyswarm2_launch,
        control_node,
        perception_node,
        gcs_node,
    ])
