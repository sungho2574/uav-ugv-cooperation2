#!/usr/bin/env python3
"""Real-hardware Crazyflie telemetry + marker/object detection.

Same /states and /detections topic contract as sim_perception_node.py, but
position comes from crazyswarm2's real backend (/cfN/pose, populated over
radio -- we don't parse radio telemetry ourselves) and detections come from
actually running a vision model on AI-deck video instead of checking a
ground-truth file. Also publishes /mission/link_status (real-only, no sim
equivalent): per-drone radio/WiFi connectivity, shown as badges in the GCS
video panel -- radio connectivity is inferred from /cfN/pose staleness since
crazyswarm2 has no explicit "connected" boolean topic, WiFi connectivity
comes straight from DroneLink's socket state.

Two interchangeable detection backends, picked per mission_map.yaml's
`detection_backend` field (see real.launch.py, which reads it and passes it
in as this node's `detection_backend` parameter):

  - "aruco" (default): ArucoBackend. Fiducial markers give a persistent,
    unambiguous ID directly, plus a metric depth via solvePnP (marker_size is
    known), so world position comes from the marker's own geometry.
  - "yolo": YoloBackend. A YOLOv8-ONNX object detector gives a class + 2D
    pixel box but no depth and no persistent identity. World position is
    instead recovered by casting a ray from the camera through the box's
    ground-contact pixel and intersecting it with the (known-altitude) ground
    plane -- see pixel_ray_to_world. Since there's no true per-object ID,
    repeated sightings of the same physical object are deduplicated by
    rounding its *computed world position* into a grid bucket (see
    _synthetic_marker_id) rather than by a real tracked identity -- two
    same-class objects closer together than `yolo.cluster_radius` will be
    merged into one record. See docs/map_configuration.md for tuning this.

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

from crazyflie_interfaces.msg import Status
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from mission_interfaces.msg import DroneState, LinkStatus, LinkStatusArray, MarkerDetection
from sensor_msgs.msg import Image

from cf_perception.yolo_detector import YoloDetector

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


def _rotate_body_to_world_yaw(v_body, yaw_rad):
    """Rotate a body-frame vector into the world frame assuming zero roll/pitch
    (the drone is level) -- only yaw is applied, about the world Z axis. Shared
    by both backends: marker_to_world (ArUco, known depth) and
    pixel_ray_to_world (YOLO, ray direction only, depth solved separately)."""
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    x_rel = cy * v_body[0] - sy * v_body[1]
    y_rel = sy * v_body[0] + cy * v_body[1]
    z_rel = v_body[2]
    return x_rel, y_rel, z_rel


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
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
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
    """ArUco path: tvec is a full metric 3D point from solvePnP (marker_size
    gives real depth), so this only needs to rotate it into world and add the
    drone's own position."""
    p_body = R_CAM_TO_BODY @ tvec
    x_rel, y_rel, z_rel = _rotate_body_to_world_yaw(p_body, yaw_rad)
    return drone_x + x_rel, drone_y + y_rel, drone_z + z_rel


def pixel_ray_to_world(u, v, camera_matrix, drone_pose, ground_z=0.0):
    """YOLO path: no known object size, so there's no metric depth from a
    single frame -- instead cast a ray from the camera through pixel (u, v)
    and intersect it with the horizontal plane z = ground_z (the object's
    known/assumed height off the floor, e.g. 0 for something lying flat).

    Returns (x, y, z) in world coordinates, or None if the ray doesn't hit the
    plane in front of the camera (e.g. it points above the horizon, which
    shouldn't normally happen at a 45 deg nose-down mount unless the drone is
    heavily pitched/rolled -- see the level-flight assumption in
    _rotate_body_to_world_yaw)."""
    drone_x, drone_y, drone_z, yaw_rad = drone_pose
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    d_body = R_CAM_TO_BODY @ d_cam
    dx, dy, dz = _rotate_body_to_world_yaw(d_body, yaw_rad)

    if abs(dz) < 1e-9:
        return None  # ray is parallel to the ground plane, no intersection
    t = (ground_z - drone_z) / dz
    if t <= 0:
        return None  # plane is behind the camera along this ray
    return drone_x + t * dx, drone_y + t * dy, ground_z


def _synthetic_marker_id(class_id, world_x, world_y, cluster_radius):
    """YOLO has no persistent per-object identity (unlike an ArUco ID), so
    stand-ins for it are minted by bucketing each detection's *computed world
    position* into a `cluster_radius`-sized grid cell -- repeat sightings of
    the same physical object land in the same bucket and dedupe the way
    control_node already dedupes ArUco marker_ids (first-seen position wins).
    This is a spatial approximation, not real tracking: two same-class objects
    closer together than cluster_radius will collide into a single record.
    Offset well clear of ArUco's 0-249 ID range and kept a positive int32."""
    bucket_x = round(world_x / cluster_radius)
    bucket_y = round(world_y / cluster_radius)
    h = (class_id * 1_000_003) ^ (bucket_x * 92_821) ^ (bucket_y * 68_927)
    return 1_000_000 + (h & 0x0FFFFFFF)


class ArucoBackend:
    """Wraps ArucoDetector into the common backend interface DroneLink uses:
    process(frame, drone_pose) draws the overlay in place and returns
    (results, raw_count) -- results is a list of {marker_id, x, y, z}
    world-coordinate detections; raw_count is how many markers were visually
    detected in this frame regardless of whether a synced drone_pose was
    available to turn them into world coordinates (see DroneLink._process_frames,
    which uses the raw_count vs len(results) gap to diagnose "GCS shows
    nothing even though the target is clearly on camera" on real hardware)."""

    def __init__(self, camera_matrix, dist_coeffs, marker_size_m):
        self.detector = ArucoDetector(camera_matrix, dist_coeffs, marker_size_m)

    def process(self, frame_bgr, drone_pose):
        corners, ids = self.detector.detect_raw(frame_bgr)
        poses = self.detector.solve_poses(corners, ids)
        self.detector.draw_overlay(frame_bgr, corners, ids, poses)

        results = []
        if drone_pose is not None:
            px, py, pz, pyaw = drone_pose
            for pose in poses:
                wx, wy, wz = marker_to_world(pose['tvec'], px, py, pz, pyaw)
                results.append({'marker_id': pose['id'], 'x': wx, 'y': wy, 'z': wz})
        return results, len(poses)


class YoloBackend:
    """Wraps YoloDetector + pixel_ray_to_world into the same interface as
    ArucoBackend (see its docstring for the raw_count/results distinction)."""

    def __init__(self, weights_path, camera_matrix, confidence_threshold,
                 nms_threshold, target_height, cluster_radius):
        self.yolo = YoloDetector(weights_path, confidence_threshold, nms_threshold)
        self.camera_matrix = camera_matrix
        self.target_height = target_height
        self.cluster_radius = cluster_radius

    def process(self, frame_bgr, drone_pose):
        detections = self.yolo.detect_raw(frame_bgr)
        self._draw_overlay(frame_bgr, detections)

        results = []
        if drone_pose is not None:
            for det in detections:
                u, v = det['ground_px']
                hit = pixel_ray_to_world(
                    u, v, self.camera_matrix, drone_pose, self.target_height)
                if hit is None:
                    continue
                wx, wy, wz = hit
                marker_id = _synthetic_marker_id(det['class_id'], wx, wy, self.cluster_radius)
                results.append({'marker_id': marker_id, 'x': wx, 'y': wy, 'z': wz})
        return results, len(detections)

    @staticmethod
    def _draw_overlay(frame_bgr, detections):
        for det in detections:
            x, y, bw, bh = [int(round(v)) for v in det['bbox']]
            cv2.rectangle(frame_bgr, (x, y), (x + bw, y + bh), (0, 200, 255), 1)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(
                frame_bgr, label, (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 200, 255), 1, cv2.LINE_AA)


class DroneLink:
    """One WiFi video connection + detection pipeline for a single drone."""

    # How stale the last /cfN/pose can be, relative to a frame's arrival time,
    # before that frame's detections are dropped instead of turned into world
    # coordinates (there's no pose to combine them with). WiFi video and radio
    # telemetry are two independent, unsynchronized network streams -- each
    # with its own real-world jitter -- so this was originally 0.2s (fine on a
    # clean local sim run) but proved too tight on real hardware: the overlay
    # box would show a clearly-detected target on the video panel while
    # /detections (and therefore the GCS 3D view / marker list) stayed empty,
    # because almost every frame missed the sync window. Widened to 0.5s: at
    # typical COVERING speeds (cruise_speed ~0.05-0.1 m/s) that's still only a
    # few cm of position lag, which is small next to the ray/solvePnP error
    # this pipeline already has.
    POSE_SYNC_TOLERANCE_SEC = 0.5
    # /cfN/pose streams at firmware_logging's configured pose frequency
    # (10Hz in crazyflies.yaml) whenever the radio link is actually up --
    # crazyswarm2 doesn't publish an explicit "radio connected" boolean, so
    # this node infers it from how stale the last received pose is instead.
    RADIO_STALE_SEC = 1.0
    # Minimum gap between "detected but couldn't localize" warnings, so a
    # steady stream of dropped frames doesn't spam the log.
    POSE_DROP_WARN_INTERVAL_SEC = 3.0

    def __init__(self, node, drone_id, wifi_ip, wifi_port, backend, bridge):
        self.node = node
        self.drone_id = drone_id
        self.wifi_ip = wifi_ip
        self.wifi_port = wifi_port
        self.backend = backend  # ArucoBackend or YoloBackend, see module docstring
        self.bridge = bridge
        self.latest_pose = None  # (x, y, z, yaw, wall_clock_stamp_sec)
        self.wifi_connected = False  # set from the background _run thread
        self.battery_voltage = 0.0  # volts; 0.0 = no /status reading yet
        self._last_pose_drop_warn = 0.0
        self.image_pub = node.create_publisher(Image, f'/{drone_id}/image_raw', 5)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update_pose(self, x, y, z, yaw, stamp_sec):
        self.latest_pose = (x, y, z, yaw, stamp_sec)

    def update_battery(self, voltage):
        self.battery_voltage = voltage

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

            # Overlay is drawn in place by process() regardless of whether a
            # given detection is also good enough to emit a /detections
            # world-coordinate report below, so the published image always
            # reflects everything the backend saw on camera.
            drone_pose = None
            pose_gap = None
            if self.latest_pose is not None:
                px, py, pz, pyaw, pose_stamp = self.latest_pose
                pose_gap = abs(now - pose_stamp)
                if pose_gap < self.POSE_SYNC_TOLERANCE_SEC:
                    drone_pose = (px, py, pz, pyaw)
            results, raw_count = self.backend.process(frame, drone_pose)

            if raw_count > 0 and not results and now - self._last_pose_drop_warn > self.POSE_DROP_WARN_INTERVAL_SEC:
                self._last_pose_drop_warn = now
                if self.latest_pose is None:
                    reason = 'no /cfN/pose received yet'
                else:
                    reason = f'last pose is {pose_gap:.2f}s old (limit {self.POSE_SYNC_TOLERANCE_SEC}s)'
                self.node.get_logger().warn(
                    f'{self.drone_id}: target visible on camera ({raw_count} detection(s)) but '
                    f'not reported to GCS -- {reason}. Check the radio link/pose rate; this will '
                    'not show up in /mission/markers or the GCS 3D view until pose syncs.')

            for result in results:
                msg = MarkerDetection()
                msg.header.frame_id = 'map'
                msg.header.stamp = self.node.get_clock().now().to_msg()
                msg.drone_id = self.drone_id
                msg.marker_id = result['marker_id']
                msg.position.x = result['x']
                msg.position.y = result['y']
                msg.position.z = result['z']
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
        # detection_backend: 'aruco' (default) or 'yolo' -- see module
        # docstring. Set via mission_map.yaml's `detection_backend` field,
        # wired through by real.launch.py.
        self.declare_parameter('detection_backend', 'aruco')
        self.declare_parameter('yolo_weights_path', '')
        self.declare_parameter('yolo_confidence_threshold', 0.5)
        self.declare_parameter('yolo_nms_threshold', 0.45)
        self.declare_parameter('yolo_target_height', 0.0)
        self.declare_parameter('yolo_cluster_radius', 0.5)

        self.drone_ids = list(self.get_parameter('drone_ids').value)
        wifi_ips = list(self.get_parameter('wifi_ips').value)
        wifi_port = self.get_parameter('wifi_port').value
        marker_size = self.get_parameter('marker_size').value
        detection_backend = self.get_parameter('detection_backend').value

        if len(wifi_ips) != len(self.drone_ids):
            raise RuntimeError('wifi_ips must have one entry per drone_id, same order')

        camera_matrix, dist_coeffs = self._load_intrinsics(
            self.get_parameter('camera_intrinsics_path').value)

        # A fresh backend instance per drone (not one shared across all of
        # them) -- each DroneLink runs its own thread, and neither
        # cv2.aruco.ArucoDetector nor a cv2.dnn.Net is guaranteed safe to call
        # concurrently from multiple threads on the same instance.
        def build_backend():
            if detection_backend == 'yolo':
                weights_path = self.get_parameter('yolo_weights_path').value
                if not weights_path:
                    raise RuntimeError(
                        "detection_backend is 'yolo' but yolo_weights_path is empty -- "
                        'set mission_map.yaml\'s yolo.weights_path to an exported .onnx '
                        'file (see docs/map_configuration.md)')
                return YoloBackend(
                    weights_path, camera_matrix,
                    self.get_parameter('yolo_confidence_threshold').value,
                    self.get_parameter('yolo_nms_threshold').value,
                    self.get_parameter('yolo_target_height').value,
                    self.get_parameter('yolo_cluster_radius').value)
            elif detection_backend == 'aruco':
                return ArucoBackend(camera_matrix, dist_coeffs, marker_size)
            else:
                raise RuntimeError(
                    f"unknown detection_backend '{detection_backend}' -- "
                    "must be 'aruco' or 'yolo'")

        bridge = CvBridge()

        self.states_pub = self.create_publisher(DroneState, '/states', 10)
        self.detections_pub = self.create_publisher(MarkerDetection, '/detections', 10)
        self.link_status_pub = self.create_publisher(
            LinkStatusArray, '/mission/link_status', 10)

        self.get_logger().info(f'real_perception_node using detection_backend={detection_backend}')

        self.links = {}
        self.pose_subs = []
        self.status_subs = []
        for drone_id, wifi_ip in zip(self.drone_ids, wifi_ips):
            self.links[drone_id] = DroneLink(
                self, drone_id, wifi_ip, wifi_port, build_backend(), bridge)
            self.pose_subs.append(self.create_subscription(
                PoseStamped, f'/{drone_id}/pose',
                self._make_pose_callback(drone_id), 10))
            # crazyswarm2's real backend already logs pm.vbatMV at ~1Hz (see
            # crazyflies.yaml's `all.firmware_logging.default_topics.status`)
            # and publishes it here as battery_voltage (volts) -- no firmware
            # or config changes needed, just subscribe.
            self.status_subs.append(self.create_subscription(
                Status, f'/{drone_id}/status',
                self._make_status_callback(drone_id), 10))

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

    def _make_status_callback(self, drone_id):
        def callback(msg: Status):
            self.links[drone_id].update_battery(msg.battery_voltage)
        return callback

    def _publish_link_status(self):
        now = time.time()
        array = LinkStatusArray()
        for drone_id, link in self.links.items():
            array.status.append(LinkStatus(
                drone_id=drone_id,
                radio_connected=link.radio_connected(now),
                wifi_connected=link.wifi_connected,
                battery_voltage=link.battery_voltage,
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
