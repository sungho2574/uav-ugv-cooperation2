#!/usr/bin/env python3
"""GCS dashboard node: Flask + Three.js front end fed by ROS2 topics.

Runs an rclpy node on a background thread that maintains a thread-locked
SharedState snapshot (drone poses, marker detections, zone/coverage-path
plan, mission phase, latest camera frames), while Flask (main thread) serves
that snapshot over a small REST API the browser polls -- same overall shape
as uav-ugv-cooperation/dashboard/dashboard_aruco.py, but with a 3D Three.js
scene instead of a 2D SVG map.
"""
import os
import threading
import time

import cv2
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from flask import Flask, Response, jsonify, render_template
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from mission_interfaces.msg import (
    CoveragePathArray, DroneProgressArray, DroneState, MarkerDetection, ZoneAssignmentArray)
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
)

DRONE_COLORS = {'cf1': '#ff5555', 'cf2': '#55aaff', 'cf3': '#55dd77'}
DEFAULT_COLOR = '#cccccc'


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self.drones = {}    # drone_id -> {x, y, z, yaw}
        self.markers = {}   # marker_id -> {x, y, z}
        self.zones = []     # [{drone_id, color, polygons: [[[x, y], ...]]}]
        self.paths = {}     # drone_id -> [[x, y, z], ...]
        self.progress = {}  # drone_id -> {waypoint_index, total_waypoints}
        self.mission_state = 'UNKNOWN'
        self.frames = {}    # drone_id -> jpeg bytes

    def update_drone(self, drone_id, x, y, z, yaw):
        with self._lock:
            self.drones[drone_id] = {'x': x, 'y': y, 'z': z, 'yaw': yaw}

    def update_marker(self, marker_id, x, y, z):
        with self._lock:
            self.markers.setdefault(marker_id, {'x': x, 'y': y, 'z': z})

    def set_zones(self, zones):
        with self._lock:
            self.zones = zones

    def set_paths(self, paths):
        with self._lock:
            self.paths = paths

    def set_progress(self, progress):
        with self._lock:
            self.progress = progress

    def set_mission_state(self, text):
        with self._lock:
            self.mission_state = text

    def update_frame(self, drone_id, jpeg_bytes):
        with self._lock:
            self.frames[drone_id] = jpeg_bytes

    def get_frame(self, drone_id):
        with self._lock:
            return self.frames.get(drone_id)

    def snapshot(self):
        with self._lock:
            return {
                'drones': [{'id': did, **s} for did, s in self.drones.items()],
                'markers': [{'id': mid, **m} for mid, m in self.markers.items()],
                'zones': list(self.zones),
                'paths': dict(self.paths),
                'progress': dict(self.progress),
                'mission_state': self.mission_state,
            }


class GcsNode(Node):

    def __init__(self, shared):
        super().__init__('gcs_node')
        self.shared = shared
        self.bridge = CvBridge()

        self.declare_parameter('drone_ids', ['cf1', 'cf2', 'cf3'])
        self.declare_parameter('port', 5000)
        self.declare_parameter('mission_map_path', '')
        self.declare_parameter('true_markers_path', '')
        self.drone_ids = list(self.get_parameter('drone_ids').value)
        self.port = self.get_parameter('port').value
        self.map_info = self._load_map_info(self.get_parameter('mission_map_path').value)
        self.all_markers = self._load_all_markers(self.get_parameter('true_markers_path').value)

        self.create_subscription(DroneState, '/states', self._on_state, 20)
        self.create_subscription(MarkerDetection, '/detections', self._on_detection, 20)
        self.create_subscription(
            ZoneAssignmentArray, '/mission/zones', self._on_zones, LATCHED_QOS)
        self.create_subscription(
            CoveragePathArray, '/mission/coverage_paths', self._on_paths, LATCHED_QOS)
        self.create_subscription(DroneProgressArray, '/mission/progress', self._on_progress, 10)
        self.create_subscription(String, '/mission/state', self._on_mission_state, 10)

        for drone_id in self.drone_ids:
            self.create_subscription(
                Image, f'/{drone_id}/image_raw',
                self._make_image_callback(drone_id), 2)

        self.start_client = self.create_client(Trigger, '/mission/start')

    def _load_map_info(self, path):
        """mission_map.yaml is known in advance (unlike true_markers.yaml), so gcs_dashboard
        just reads boundary/dead_zones straight from it rather than needing control_node to
        republish static map geometry over a topic."""
        if not path:
            self.get_logger().warn('mission_map_path not set, boundary/dead_zones will not render')
            return {'boundary': [], 'dead_zones': [], 'coverage_line_spacing': 0.5}
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return {
            'boundary': data.get('boundary', []),
            'dead_zones': [dz.get('points', []) for dz in data.get('dead_zones', [])],
            'coverage_line_spacing': data.get('coverage_line_spacing', 0.5),
        }

    def _load_all_markers(self, path):
        """Sim-only debug overlay: true_markers.yaml ground truth, purely for showing
        "not found yet" marker placeholders on the ground. On real hardware this
        parameter is left unset (real.launch.py never passes it) so this list stays
        empty and the dashboard only ever shows markers once actually /detections'd --
        exactly as it should be, since real ground truth genuinely isn't known ahead
        of time. control_node never sees this list either way, so mission logic
        itself is never able to "cheat" off of it."""
        if not path:
            return []
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return [
            {'id': m['id'], 'x': m['x'], 'y': m['y'], 'z': m.get('z', 0.0)}
            for m in data.get('markers', [])
        ]

    def _on_state(self, msg: DroneState):
        self.shared.update_drone(
            msg.drone_id, msg.position.x, msg.position.y, msg.position.z, msg.yaw)

    def _on_detection(self, msg: MarkerDetection):
        self.shared.update_marker(msg.marker_id, msg.position.x, msg.position.y, msg.position.z)

    def _on_zones(self, msg: ZoneAssignmentArray):
        zones = []
        for za in msg.zones:
            polygons = [[[p.x, p.y] for p in poly.points] for poly in za.polygons]
            zones.append({
                'drone_id': za.drone_id,
                'color': DRONE_COLORS.get(za.drone_id, DEFAULT_COLOR),
                'polygons': polygons,
            })
        self.shared.set_zones(zones)

    def _on_paths(self, msg: CoveragePathArray):
        paths = {}
        for cp in msg.paths:
            paths[cp.drone_id] = [
                [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
                for pose in cp.path.poses
            ]
        self.shared.set_paths(paths)

    def _on_progress(self, msg: DroneProgressArray):
        progress = {
            p.drone_id: {'waypoint_index': p.waypoint_index, 'total_waypoints': p.total_waypoints}
            for p in msg.progress
        }
        self.shared.set_progress(progress)

    def _on_mission_state(self, msg: String):
        self.shared.set_mission_state(msg.data)

    def _make_image_callback(self, drone_id):
        def callback(msg: Image):
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                self.shared.update_frame(drone_id, buf.tobytes())
        return callback

    def request_mission_start(self):
        """Called from the Flask thread -- do NOT spin here, the background
        thread is already spinning this node; just poll the future instead."""
        if not self.start_client.service_is_ready():
            return False, 'mission/start service not available'
        future = self.start_client.call_async(Trigger.Request())
        deadline = time.time() + 2.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.01)
        if future.done() and future.result() is not None:
            return future.result().success, future.result().message
        return False, 'mission/start call timed out'


shared_state = SharedState()
ros_node = None

share_dir = get_package_share_directory('gcs_dashboard')
app = Flask(
    __name__,
    template_folder=os.path.join(share_dir, 'templates'),
    static_folder=os.path.join(share_dir, 'static'),
)


@app.route('/')
def index():
    return render_template('index.html', drone_ids=ros_node.drone_ids if ros_node else [])


@app.route('/api/state')
def api_state():
    return jsonify(shared_state.snapshot())


@app.route('/api/map')
def api_map():
    return jsonify(ros_node.map_info if ros_node else {'boundary': [], 'dead_zones': []})


@app.route('/api/all_markers')
def api_all_markers():
    """Sim-only ground-truth marker list (empty on real hardware). See
    GcsNode._load_all_markers for why this can never leak into control_node."""
    return jsonify(ros_node.all_markers if ros_node else [])


@app.route('/api/frame/<drone_id>')
def api_frame(drone_id):
    jpeg = shared_state.get_frame(drone_id)
    if jpeg is None:
        return '', 204
    return Response(jpeg, mimetype='image/jpeg')


@app.route('/api/mission/start', methods=['POST'])
def api_mission_start():
    success, message = ros_node.request_mission_start()
    return jsonify({'success': success, 'message': message})


def main(args=None):
    global ros_node
    rclpy.init(args=args)
    ros_node = GcsNode(shared_state)

    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    try:
        app.run(host='0.0.0.0', port=ros_node.port, threaded=True)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
