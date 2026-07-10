#!/usr/bin/env python3
"""Real-hardware Crazyflie telemetry + marker detection.

Same /states and /detections topic contract as sim_perception_node.py, but
position comes from crazyswarm2's real backend (/cfN/pose, populated over
radio -- we don't parse radio telemetry ourselves) and marker detections come
from actually running ArUco on AI-deck video instead of checking a
ground-truth file. Also publishes /mission/link_status (real-only, no sim
equivalent): per-drone radio/WiFi connectivity, shown as badges in the GCS
video panel -- radio connectivity is inferred from /cfN/pose staleness since
crazyswarm2 has no explicit "connected" boolean topic, WiFi connectivity
comes straight from DroneLink's socket state.

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
from mission_interfaces.msg import DroneState, LinkStatus, LinkStatusArray, MarkerDetection
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


def receive_frame(sock, logger=None):
    """Receive one AI-deck WiFi frame. Returns a BGR numpy array or None."""
    packet_info = rx_bytes(sock, 4)
    length, _routing, _function = struct.unpack('<HBB', packet_info)

    img_header = rx_bytes(sock, length - 2)
    magic, width, height, _depth, fmt, size = struct.unpack('<BHHBBI', img_header)
    if magic != 0xBC:
        # Framing is off (this wasn't actually an image header) -- silently
        # returning None here used to look identical to "no image yet",
        # which is exactly what made a real protocol mismatch invisible.
        if logger is not None:
            logger.warn(f'unexpected magic byte 0x{magic:02x} (expected 0xbc) -- '
                        'AI-deck WiFi stream framing looks wrong')
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
        # Deliberately NOT settimeout(None) here: a fully blocking socket
        # means a deck that connects but never actually streams anything
        # (wrong firmware/example flashed, camera not initialized, etc.)
        # hangs receive_frame forever with zero errors or logs -- the TCP
        # handshake alone was already enough to make the GCS "WiFi" badge
        # show connected, so that silent hang looked identical to "working
        # but idle". A read timeout turns that into a periodic, visible
        # reconnect-with-a-log-message instead.
        s.settimeout(5.0)
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
        self.marker_size_m = marker_size_m
        half = marker_size_m / 2.0
        self.obj_pts = np.array([
            [-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0],
        ], dtype=np.float32)

    def detect_raw(self, frame_bgr):
        """Plain marker detection (corners + ids), no pose solve yet -- used
        so the overlay can show every marker the detector sees even if its
        solvePnP later gets rejected (e.g. too far away)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        return corners, ids

    def solve_poses(self, corners, ids):
        """Per-marker solvePnP -> world-usable results, filtering out
        implausible ranges (marker seen but too far/degenerate pose)."""
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
            results.append({'id': int(ids[i][0]), 'tvec': tvec.flatten(), 'rvec': rvec})
        return results

    def draw_overlay(self, frame_bgr, corners, ids, poses):
        """Mutates frame_bgr in place: marker outline+id for every detected
        marker, plus a pose axis for whichever ones had a valid solvePnP."""
        if ids is None:
            return
        cv2.aruco.drawDetectedMarkers(frame_bgr, corners, ids)
        axis_len = self.marker_size_m / 2.0
        for pose in poses:
            cv2.drawFrameAxes(
                frame_bgr, self.camera_matrix, self.dist_coeffs,
                pose['rvec'], pose['tvec'], axis_len)


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
    # /cfN/pose streams at firmware_logging's configured pose frequency
    # (10Hz in crazyflies.yaml) whenever the radio link is actually up --
    # crazyswarm2 doesn't publish an explicit "radio connected" boolean, so
    # this node infers it from how stale the last received pose is instead.
    RADIO_STALE_SEC = 1.0

    def __init__(self, node, drone_id, wifi_ip, wifi_port, detector, bridge):
        self.node = node
        self.drone_id = drone_id
        self.wifi_ip = wifi_ip
        self.wifi_port = wifi_port
        self.detector = detector
        self.bridge = bridge
        self.latest_pose = None  # (x, y, z, yaw, wall_clock_stamp_sec)
        self.wifi_connected = False  # set from the background _run thread
        self.image_pub = node.create_publisher(Image, f'/{drone_id}/image_raw', 5)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update_pose(self, x, y, z, yaw, stamp_sec):
        self.latest_pose = (x, y, z, yaw, stamp_sec)

    def radio_connected(self, now):
        if self.latest_pose is None:
            return False
        return (now - self.latest_pose[4]) < self.RADIO_STALE_SEC

    def _run(self):
        while rclpy.ok():
            sock = try_connect(self.wifi_ip, self.wifi_port, timeout=3.0)
            if sock is None:
                self.wifi_connected = False
                time.sleep(3.0)
                continue
            self.wifi_connected = True
            self.node.get_logger().info(
                f'{self.drone_id} wifi connected to {self.wifi_ip}:{self.wifi_port}, '
                'waiting for frames...')
            try:
                self._process_frames(sock)
            except socket.timeout:
                self.node.get_logger().warn(
                    f'{self.drone_id} wifi connected but received nothing for 5s -- '
                    'TCP link is up but the AI-deck does not seem to be streaming '
                    '(wrong example/firmware flashed, camera not initialized, etc.)')
            except Exception as exc:
                self.node.get_logger().warn(f'{self.drone_id} wifi link dropped: {exc}')
            finally:
                self.wifi_connected = False
                sock.close()
            time.sleep(1.0)

    def _process_frames(self, sock):
        frame_count = 0
        while rclpy.ok():
            frame = receive_frame(sock, logger=self.node.get_logger())
            if frame is None:
                continue
            now = time.time()
            frame_count += 1
            if frame_count == 1:
                self.node.get_logger().info(
                    f'{self.drone_id} first frame received ({frame.shape[1]}x{frame.shape[0]})')
            elif frame_count % 100 == 0:
                self.node.get_logger().info(f'{self.drone_id} {frame_count} frames received so far')

            corners, ids = self.detector.detect_raw(frame)
            poses = self.detector.solve_poses(corners, ids)
            # Overlay every marker the detector saw (box + id), plus a pose
            # axis for the ones with a usable solvePnP -- drawn in place so
            # the published image always reflects what's on camera, whether
            # or not that detection was good enough to also emit a
            # /detections world-coordinate report below.
            self.detector.draw_overlay(frame, corners, ids, poses)

            if poses and self.latest_pose is not None:
                px, py, pz, pyaw, pose_stamp = self.latest_pose
                if abs(now - pose_stamp) < self.POSE_SYNC_TOLERANCE_SEC:
                    for pose in poses:
                        wx, wy, wz = marker_to_world(pose['tvec'], px, py, pz, pyaw)
                        msg = MarkerDetection()
                        msg.header.frame_id = 'map'
                        msg.header.stamp = self.node.get_clock().now().to_msg()
                        msg.drone_id = self.drone_id
                        msg.marker_id = pose['id']
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

        self.declare_parameter('drone_ids', ['cf6', 'cf7', 'cf8'])
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
        self.link_status_pub = self.create_publisher(
            LinkStatusArray, '/mission/link_status', 10)

        self.links = {}
        self.pose_subs = []
        for drone_id, wifi_ip in zip(self.drone_ids, wifi_ips):
            self.links[drone_id] = DroneLink(
                self, drone_id, wifi_ip, wifi_port, detector, bridge)
            self.pose_subs.append(self.create_subscription(
                PoseStamped, f'/{drone_id}/pose',
                self._make_pose_callback(drone_id), 10))

        # Radio connectivity is inferred from pose staleness (see
        # DroneLink.radio_connected), so it needs to be re-evaluated on a
        # timer even when no new pose arrives -- a drone that goes silent
        # should flip to "disconnected" on its own, not just wait for the
        # next pose message that may never come.
        self.create_timer(0.5, self._publish_link_status)

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

    def _publish_link_status(self):
        now = time.time()
        array = LinkStatusArray()
        for drone_id, link in self.links.items():
            array.status.append(LinkStatus(
                drone_id=drone_id,
                radio_connected=link.radio_connected(now),
                wifi_connected=link.wifi_connected,
            ))
        self.link_status_pub.publish(array)


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
