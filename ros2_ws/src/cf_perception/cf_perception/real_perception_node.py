#!/usr/bin/env python3
"""Real-hardware Crazyflie telemetry + marker detection.

Same /states and /detections topic contract as sim_perception_node.py, but
position comes from crazyswarm2's real backend (/cfN/pose, populated over
radio -- we don't parse radio telemetry ourselves) and marker detections come
from actually running ArUco on AI-deck video instead of checking a
ground-truth file.

The WiFi frame protocol (rx_bytes/receive_frame/try_connect, magic byte 0xBC)
and the camera->body rotation assumption are carried over verbatim from the
hardware-verified uav-ugv-cooperation/dashboard/dashboard_aruco.py prototype.
"""
import math
import socket
import struct
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from mission_interfaces.msg import DroneState, MarkerDetection
from sensor_msgs.msg import Image

# Reused from uav-ugv-cooperation/dashboard/dashboard_aruco.py: assumes the AI-deck
# camera is mounted pointing 45 deg nose-down from the drone's front. VERIFY against
# the actual mounting angle on the real hardware before trusting detections.
_S = math.sqrt(2) / 2
R_CAM_TO_BODY = np.array(
    [[0.0, -_S, _S],
     [-1.0, 0.0, 0.0],
     [0.0, -_S, -_S]],
    dtype=np.float64,
)


def rx_bytes(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError('socket closed')
        data.extend(chunk)
    return data


def receive_frame(sock):
    """Receive one AI-deck WiFi frame. Returns a BGR numpy array or None."""
    packet_info = rx_bytes(sock, 4)
    length, _routing, _function = struct.unpack('<HBB', packet_info)

    img_header = rx_bytes(sock, length - 2)
    magic, width, height, _depth, fmt, size = struct.unpack('<BHHBBI', img_header)
    if magic != 0xBC:
        return None

    img_stream = bytearray()
    while len(img_stream) < size:
        packet_info = rx_bytes(sock, 4)
        chunk_length, _dst, _src = struct.unpack('<HBB', packet_info)
        img_stream.extend(rx_bytes(sock, chunk_length - 2))

    if fmt == 0:  # raw bayer
        img = np.frombuffer(img_stream, dtype=np.uint8).reshape(height, width)
        return cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)

    nparr = np.frombuffer(img_stream, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def try_connect(ip, port, timeout=2.0):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((ip, port))
        s.settimeout(None)
        return s
    except OSError:
        return None


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ArucoDetector:
    def __init__(self, camera_matrix, dist_coeffs, marker_size_m):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        half = marker_size_m / 2.0
        self.obj_pts = np.array([
            [-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0],
        ], dtype=np.float32)

    def detect(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        results = []
        if ids is None:
            return results
        for i, corner in enumerate(corners):
            img_pts = corner[0].astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                self.obj_pts, img_pts, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok or np.linalg.norm(tvec) > 8.0:
                continue
            results.append({'id': int(ids[i][0]), 'tvec': tvec.flatten()})
        return results


def marker_to_world(tvec, drone_x, drone_y, drone_z, yaw_rad):
    p_body = R_CAM_TO_BODY @ tvec
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    x_rel = cy * p_body[0] - sy * p_body[1]
    y_rel = sy * p_body[0] + cy * p_body[1]
    z_rel = p_body[2]
    return drone_x + x_rel, drone_y + y_rel, drone_z + z_rel


class DroneLink:
    """One WiFi video connection + ArUco pipeline for a single drone."""

    POSE_SYNC_TOLERANCE_SEC = 0.2

    def __init__(self, node, drone_id, wifi_ip, wifi_port, detector, bridge):
        self.node = node
        self.drone_id = drone_id
        self.wifi_ip = wifi_ip
        self.wifi_port = wifi_port
        self.detector = detector
        self.bridge = bridge
        self.latest_pose = None  # (x, y, z, yaw, wall_clock_stamp_sec)
        self.image_pub = node.create_publisher(Image, f'/{drone_id}/image_raw', 5)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update_pose(self, x, y, z, yaw, stamp_sec):
        self.latest_pose = (x, y, z, yaw, stamp_sec)

    def _run(self):
        while rclpy.ok():
            sock = try_connect(self.wifi_ip, self.wifi_port, timeout=3.0)
            if sock is None:
                time.sleep(3.0)
                continue
            try:
                self._process_frames(sock)
            except Exception as exc:
                self.node.get_logger().warn(f'{self.drone_id} wifi link dropped: {exc}')
            finally:
                sock.close()
            time.sleep(1.0)

    def _process_frames(self, sock):
        while rclpy.ok():
            frame = receive_frame(sock)
            if frame is None:
                continue
            now = time.time()

            detections = self.detector.detect(frame)
            if detections and self.latest_pose is not None:
                px, py, pz, pyaw, pose_stamp = self.latest_pose
                if abs(now - pose_stamp) < self.POSE_SYNC_TOLERANCE_SEC:
                    for det in detections:
                        wx, wy, wz = marker_to_world(det['tvec'], px, py, pz, pyaw)
                        msg = MarkerDetection()
                        msg.header.frame_id = 'map'
                        msg.header.stamp = self.node.get_clock().now().to_msg()
                        msg.drone_id = self.drone_id
                        msg.marker_id = det['id']
                        msg.position.x = wx
                        msg.position.y = wy
                        msg.position.z = wz
                        self.node.detections_pub.publish(msg)

            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img_msg.header.frame_id = self.drone_id
            img_msg.header.stamp = self.node.get_clock().now().to_msg()
            self.image_pub.publish(img_msg)


class RealPerceptionNode(Node):

    def __init__(self):
        super().__init__('real_perception_node')

        self.declare_parameter('drone_ids', ['cf1', 'cf2', 'cf3'])
        self.declare_parameter('wifi_ips', [''])
        self.declare_parameter('wifi_port', 5000)
        self.declare_parameter('marker_size', 0.14)
        self.declare_parameter('camera_intrinsics_path', '')

        self.drone_ids = list(self.get_parameter('drone_ids').value)
        wifi_ips = list(self.get_parameter('wifi_ips').value)
        wifi_port = self.get_parameter('wifi_port').value
        marker_size = self.get_parameter('marker_size').value

        if len(wifi_ips) != len(self.drone_ids):
            raise RuntimeError('wifi_ips must have one entry per drone_id, same order')

        camera_matrix, dist_coeffs = self._load_intrinsics(
            self.get_parameter('camera_intrinsics_path').value)
        detector = ArucoDetector(camera_matrix, dist_coeffs, marker_size)
        bridge = CvBridge()

        self.states_pub = self.create_publisher(DroneState, '/states', 10)
        self.detections_pub = self.create_publisher(MarkerDetection, '/detections', 10)

        self.links = {}
        self.pose_subs = []
        for drone_id, wifi_ip in zip(self.drone_ids, wifi_ips):
            self.links[drone_id] = DroneLink(
                self, drone_id, wifi_ip, wifi_port, detector, bridge)
            self.pose_subs.append(self.create_subscription(
                PoseStamped, f'/{drone_id}/pose',
                self._make_pose_callback(drone_id), 10))

        self.get_logger().info(
            f'real_perception_node streaming from {list(zip(self.drone_ids, wifi_ips))}')

    def _load_intrinsics(self, path):
        if not path:
            self.get_logger().warn(
                'camera_intrinsics_path not set, using uncalibrated placeholder '
                'intrinsics (324x244 AI-deck defaults) -- calibrate before trusting '
                'detections')
            camera_matrix = np.array(
                [[320.0, 0.0, 162.0], [0.0, 320.0, 122.0], [0.0, 0.0, 1.0]],
                dtype=np.float32)
            dist_coeffs = np.zeros((5, 1), dtype=np.float32)
            return camera_matrix, dist_coeffs
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        camera_matrix = np.array(data['camera_matrix'], dtype=np.float32)
        dist_coeffs = np.array(data['dist_coeffs'], dtype=np.float32)
        return camera_matrix, dist_coeffs

    def _make_pose_callback(self, drone_id):
        def callback(msg: PoseStamped):
            x = msg.pose.position.x
            y = msg.pose.position.y
            z = msg.pose.position.z
            yaw = quat_to_yaw(msg.pose.orientation)
            stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.links[drone_id].update_pose(x, y, z, yaw, stamp_sec)

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
        return callback


def main(args=None):
    rclpy.init(args=args)
    node = RealPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
