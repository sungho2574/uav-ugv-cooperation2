#!/usr/bin/env python3
"""Sim-mode Crazyflie telemetry + marker-detection stand-in.

Replaces the camera pipeline used on real hardware: subscribes to each
Crazyflie's /cfN/pose (published by crazyswarm2's sim backend, identical
topic to the real backend) and republishes it as /states. Instead of running
ArUco detection on video, it checks the true (ground-truth) marker positions
loaded from true_markers.yaml -- a file that ONLY this node reads -- and
publishes a /detections message once a drone's live position lands inside a
marker's grid cell. control_node and gcs_dashboard never see the ground
truth file, so the search is still "blind" from their point of view, matching
what would happen with a real camera.
"""
import math

import rclpy
import yaml
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from mission_interfaces.msg import DroneState, MarkerDetection


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class SimPerceptionNode(Node):

    def __init__(self):
        super().__init__('sim_perception_node')

        self.declare_parameter('drone_ids', ['cf1', 'cf2', 'cf3'])
        self.declare_parameter('true_markers_path', '')
        self.declare_parameter('mission_map_path', '')

        self.drone_ids = list(self.get_parameter('drone_ids').value)

        mission_map_path = self.get_parameter('mission_map_path').value
        if not mission_map_path:
            raise RuntimeError('sim_perception_node requires the mission_map_path parameter')
        with open(mission_map_path, 'r') as f:
            mission_map = yaml.safe_load(f)
        self.grid_resolution = float(mission_map['grid_resolution'])
        self.detect_radius = self.grid_resolution / 2.0

        true_markers_path = self.get_parameter('true_markers_path').value
        if not true_markers_path:
            raise RuntimeError(
                'sim_perception_node requires the true_markers_path parameter')
        with open(true_markers_path, 'r') as f:
            data = yaml.safe_load(f)
        self.true_markers = data.get('markers', [])
        self.undetected_ids = {m['id'] for m in self.true_markers}

        self.states_pub = self.create_publisher(DroneState, '/states', 10)
        self.detections_pub = self.create_publisher(MarkerDetection, '/detections', 10)

        self.pose_subs = []
        for drone_id in self.drone_ids:
            sub = self.create_subscription(
                PoseStamped, f'/{drone_id}/pose',
                self._make_pose_callback(drone_id), 10)
            self.pose_subs.append(sub)

        self.get_logger().info(
            f'sim_perception_node watching {self.drone_ids}, '
            f'{len(self.true_markers)} ground-truth markers loaded')

    def _make_pose_callback(self, drone_id):
        def callback(msg: PoseStamped):
            self._on_pose(drone_id, msg)
        return callback

    def _on_pose(self, drone_id, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        yaw = quat_to_yaw(msg.pose.orientation)

        state = DroneState()
        state.header = msg.header
        if not state.header.frame_id:
            state.header.frame_id = 'map'
        state.drone_id = drone_id
        state.position.x = x
        state.position.y = y
        state.position.z = z
        state.yaw = yaw
        self.states_pub.publish(state)

        for marker in self.true_markers:
            marker_id = marker['id']
            if marker_id not in self.undetected_ids:
                continue
            dist = math.hypot(x - marker['x'], y - marker['y'])
            if dist <= self.detect_radius:
                self.undetected_ids.discard(marker_id)
                detection = MarkerDetection()
                detection.header = state.header
                detection.drone_id = drone_id
                detection.marker_id = marker_id
                detection.position.x = float(marker['x'])
                detection.position.y = float(marker['y'])
                detection.position.z = float(marker.get('z', 0.0))
                self.detections_pub.publish(detection)
                self.get_logger().info(
                    f'{drone_id} reached marker {marker_id} at '
                    f'({marker["x"]:.2f}, {marker["y"]:.2f})')


def main(args=None):
    rclpy.init(args=args)
    node = SimPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
