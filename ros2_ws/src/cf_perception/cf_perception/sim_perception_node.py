#!/usr/bin/env python3
"""Sim-mode Crazyflie telemetry + marker-detection stand-in.

Replaces the camera pipeline used on real hardware. IMPORTANT: crazyswarm2's
*sim* backend (crazyflie_sim/crazyflie_server.py) does NOT publish /cfN/pose
at all -- that topic only exists on the real/cflib backend (driven by
firmware_logging). The sim backend's default "rviz" visualization plugin
instead broadcasts a world->cfN transform on /tf on every physics step. So
this node looks up that transform via tf2 (on a timer) instead of
subscribing to /cfN/pose. Republishes as /states either way, so downstream
nodes (control_node, gcs_dashboard) don't need to know the difference.

Instead of running ArUco detection on video, it checks the true (ground-truth)
marker positions loaded from true_markers.yaml -- a file that ONLY this node
reads -- and publishes a /detections message once a drone's live position
lands inside a marker's grid cell. control_node and gcs_dashboard never see
the ground truth file, so the search is still "blind" from their point of
view, matching what would happen with a real camera.
"""
import math

import rclpy
import yaml
from rclpy.node import Node
from rclpy.time import Time

from tf2_ros import ConnectivityException, ExtrapolationException, LookupException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from mission_interfaces.msg import DroneState, MarkerDetection


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class SimPerceptionNode(Node):

    def __init__(self):
        super().__init__('sim_perception_node')

        self.declare_parameter('drone_ids', ['cf6', 'cf7', 'cf8'])
        self.declare_parameter('true_markers_path', '')
        self.declare_parameter('mission_map_path', '')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('poll_rate_hz', 10.0)

        self.drone_ids = list(self.get_parameter('drone_ids').value)
        self.world_frame = self.get_parameter('world_frame').value

        mission_map_path = self.get_parameter('mission_map_path').value
        if not mission_map_path:
            raise RuntimeError('sim_perception_node requires the mission_map_path parameter')
        with open(mission_map_path, 'r') as f:
            mission_map = yaml.safe_load(f)
        # control_node's coverage plan visits the *center* of each
        # coverage_line_spacing x line_spacing cell (see path_planning.py), so a
        # marker anywhere inside that cell -- worst case, right in a corner --
        # must still count as "reached" once the drone visits the cell center.
        # Corner distance from a cell's center is line_spacing * sqrt(2) / 2;
        # using grid_resolution here (a much finer, unrelated value) was a
        # leftover from before the coverage plan moved to cell-based waypoints
        # and made the detection radius far too small to ever trigger.
        line_spacing = float(mission_map['coverage_line_spacing'])
        self.detect_radius = line_spacing * math.sqrt(2) / 2.0

        true_markers_path = self.get_parameter('true_markers_path').value
        if not true_markers_path:
            raise RuntimeError(
                'sim_perception_node requires the true_markers_path parameter')
        with open(true_markers_path, 'r') as f:
            data = yaml.safe_load(f)
        # `or []` (not just `.get(..., [])`): a bare `markers:` key with
        # nothing under it parses to None in YAML, which the dict default
        # doesn't catch -- same gotcha as mission_map.yaml's dead_zones.
        self.true_markers = data.get('markers') or []
        self.undetected_ids = {m['id'] for m in self.true_markers}

        self.states_pub = self.create_publisher(DroneState, '/states', 10)
        self.detections_pub = self.create_publisher(MarkerDetection, '/detections', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        poll_rate_hz = self.get_parameter('poll_rate_hz').value
        self.create_timer(1.0 / poll_rate_hz, self._on_tf_poll)

        self.get_logger().info(
            f'sim_perception_node polling tf {self.world_frame}->{self.drone_ids}, '
            f'{len(self.true_markers)} ground-truth markers loaded')

    def _on_tf_poll(self):
        for drone_id in self.drone_ids:
            try:
                tf = self.tf_buffer.lookup_transform(self.world_frame, drone_id, Time())
            except (LookupException, ConnectivityException, ExtrapolationException):
                continue
            self._on_transform(drone_id, tf)

    def _on_transform(self, drone_id, tf):
        x = tf.transform.translation.x
        y = tf.transform.translation.y
        z = tf.transform.translation.z
        yaw = quat_to_yaw(tf.transform.rotation)

        state = DroneState()
        state.header.stamp = tf.header.stamp
        state.header.frame_id = self.world_frame
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
