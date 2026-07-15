from collections import deque
import math
import socket
import struct
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image

from crazyflie_interfaces.msg import Status
from mission_interfaces.msg import (
    DroneState,
    LinkStatus,
    LinkStatusArray,
    MarkerDetection,
)

from cf_perception.yolo_detector import YoloDetector


def camera_to_body_rotation(pitch_degrees):
    """AI-deck optical frame (right, down, forward) to body frame."""
    angle = math.radians(float(pitch_degrees))
    sine = math.sin(angle)
    cosine = math.cos(angle)
    return np.array(
        [
            [0.0, -sine, cosine],
            [-1.0, 0.0, 0.0],
            [0.0, -cosine, -sine],
        ],
        dtype=np.float64,
    )


def quaternion_to_rotation_matrix(quaternion):
    x, y, z, w = [float(value) for value in quaternion]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-9:
        return np.eye(3, dtype=np.float64)

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def quaternion_to_yaw(message_quaternion):
    return math.atan2(
        2.0 * (
            message_quaternion.w * message_quaternion.z
            + message_quaternion.x * message_quaternion.y
        ),
        1.0 - 2.0 * (
            message_quaternion.y * message_quaternion.y
            + message_quaternion.z * message_quaternion.z
        ),
    )


def quaternion_slerp(first, second, ratio):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= max(np.linalg.norm(first), 1.0e-12)
    second /= max(np.linalg.norm(second), 1.0e-12)

    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot

    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = first + float(ratio) * (second - first)
        return result / max(np.linalg.norm(result), 1.0e-12)

    theta = math.acos(dot)
    sine = math.sin(theta)
    first_weight = math.sin((1.0 - ratio) * theta) / sine
    second_weight = math.sin(ratio * theta) / sine
    return first_weight * first + second_weight * second


class DronePose:
    __slots__ = ('x', 'y', 'z', 'rotation_world_from_body')

    def __init__(self, x, y, z, quaternion):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.rotation_world_from_body = (
            quaternion_to_rotation_matrix(quaternion)
        )


class PoseSample:
    __slots__ = ('timestamp', 'position', 'quaternion')

    def __init__(self, timestamp, position, quaternion):
        self.timestamp = float(timestamp)
        self.position = np.asarray(
            position, dtype=np.float64)
        self.quaternion = np.asarray(
            quaternion, dtype=np.float64)


def receive_exact(sock, byte_count):
    data = bytearray()
    while len(data) < int(byte_count):
        chunk = sock.recv(int(byte_count) - len(data))
        if not chunk:
            raise ConnectionError('AI-deck socket closed')
        data.extend(chunk)
    return bytes(data)


def receive_frame(sock, logger=None):
    """Read the Bitcraze AI-deck CPX image framing."""
    packet_info = receive_exact(sock, 4)
    length, _routing, _function = struct.unpack(
        '<HBB', packet_info)
    if length < 2:
        raise ValueError(
            f'invalid AI-deck header length {length}')

    image_header = receive_exact(sock, length - 2)
    if len(image_header) != 11:
        raise ValueError(
            f'unexpected image header size '
            f'{len(image_header)}')

    (
        magic,
        width,
        height,
        _depth,
        image_format,
        image_size,
    ) = struct.unpack('<BHHBBI', image_header)

    if magic != 0xBC:
        if logger is not None:
            logger.warn(
                f'unexpected image magic 0x{magic:02x}')
        return None

    image_stream = bytearray()
    while len(image_stream) < image_size:
        chunk_header = receive_exact(sock, 4)
        chunk_length, _destination, _source = struct.unpack(
            '<HBB', chunk_header)
        if chunk_length < 2:
            raise ValueError(
                f'invalid AI-deck chunk length {chunk_length}')
        image_stream.extend(
            receive_exact(sock, chunk_length - 2))

    image_stream = image_stream[:image_size]
    if image_format == 0:
        expected = int(width) * int(height)
        if len(image_stream) != expected:
            raise ValueError(
                f'raw image size {len(image_stream)} '
                f'!= expected {expected}')
        bayer = np.frombuffer(
            image_stream, dtype=np.uint8
        ).reshape(height, width)
        return cv2.cvtColor(
            bayer, cv2.COLOR_BayerBG2BGR)

    encoded = np.frombuffer(
        image_stream, dtype=np.uint8)
    image = cv2.imdecode(
        encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            'OpenCV could not decode AI-deck frame')
    return image


def try_connect(ip_address, port):
    try:
        sock = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_RCVBUF,
            1 << 20,
        )
        sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_KEEPALIVE,
            1,
        )
        sock.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1,
        )
        sock.settimeout(3.0)
        sock.connect((str(ip_address), int(port)))
        sock.settimeout(10.0)
        return sock
    except OSError:
        return None


def pixel_ray_to_world(
    u,
    v,
    camera_matrix,
    distortion,
    drone_pose,
    rotation_body_from_camera,
    ground_z,
    minimum_downward_ray,
    maximum_ground_range,
):
    """Build both a ground-intersection estimate and a horizontal bearing ray.

    The previous implementation used only the single-frame ground intersection.
    That range estimate is very sensitive to camera pitch/intrinsic error.  The
    horizontal bearing is much more stable and is later fused across multiple
    drone poses with a RANSAC line-intersection estimator.
    """
    pixel = np.array(
        [[[float(u), float(v)]]],
        dtype=np.float64,
    )
    normalized = cv2.undistortPoints(
        pixel,
        camera_matrix.astype(np.float64),
        distortion.astype(np.float64),
    )[0, 0]

    direction_camera = np.array(
        [normalized[0], normalized[1], 1.0],
        dtype=np.float64,
    )
    direction_body = (
        rotation_body_from_camera @ direction_camera)
    direction_world = (
        drone_pose.rotation_world_from_body
        @ direction_body
    )

    horizontal = np.asarray(
        direction_world[:2], dtype=np.float64)
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm < 1.0e-8:
        return None
    horizontal /= horizontal_norm

    vertical = float(direction_world[2])
    if vertical >= -abs(float(minimum_downward_ray)):
        return None

    scale = (
        float(ground_z) - drone_pose.z
    ) / vertical
    if scale <= 0.0:
        return None

    delta_x = scale * float(direction_world[0])
    delta_y = scale * float(direction_world[1])
    ground_range = math.hypot(delta_x, delta_y)
    if ground_range > float(maximum_ground_range):
        return None

    return {
        'x': drone_pose.x + delta_x,
        'y': drone_pose.y + delta_y,
        'z': float(ground_z),
        'ground_range': ground_range,
        'ray_origin_x': drone_pose.x,
        'ray_origin_y': drone_pose.y,
        'ray_direction_x': float(horizontal[0]),
        'ray_direction_y': float(horizontal[1]),
    }


def bbox_iou(first, second):
    if first is None or second is None:
        return 0.0
    first_x, first_y, first_w, first_h = first
    second_x, second_y, second_w, second_h = second
    first_x2 = first_x + first_w
    first_y2 = first_y + first_h
    second_x2 = second_x + second_w
    second_y2 = second_y + second_h

    width = max(
        0.0,
        min(first_x2, second_x2)
        - max(first_x, second_x),
    )
    height = max(
        0.0,
        min(first_y2, second_y2)
        - max(first_y, second_y),
    )
    intersection = width * height
    union = (
        first_w * first_h
        + second_w * second_h
        - intersection
    )
    if union <= 1.0e-9:
        return 0.0
    return intersection / union


def _cross_2d(first, second):
    return (
        float(first[0]) * float(second[1])
        - float(first[1]) * float(second[0])
    )


def _point_to_ray_distance(point, origin, direction):
    point = np.asarray(point, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    delta = point - origin
    along = float(np.dot(delta, direction))
    perpendicular = abs(_cross_2d(direction, delta))
    return perpendicular, along


def robust_bearing_intersection(
    samples,
    inlier_threshold=0.22,
    minimum_parallax_deg=5.0,
    minimum_baseline=0.20,
):
    """Estimate a static XY landmark from multiple camera bearing rays.

    This is a compact bearing-only analogue of the robust multi-view landmark
    association used in established tracking systems: candidate intersections
    are generated from ray pairs, scored with RANSAC, and refined by weighted
    least squares over inlier bearing lines.
    """
    rays = []
    for sample in samples:
        origin = np.array(
            [sample[4], sample[5]], dtype=np.float64)
        direction = np.array(
            [sample[6], sample[7]], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1.0e-8:
            continue
        direction /= norm
        rays.append((origin, direction))

    if len(rays) < 3:
        return None

    # Bound RANSAC cost on Jetson.  Preserve the full temporal span instead of
    # taking only the newest rays, since parallax is what makes triangulation
    # observable.
    if len(rays) > 40:
        indices = np.linspace(
            0, len(rays) - 1, 40,
            dtype=np.int64,
        )
        rays = [rays[int(index)] for index in indices]

    origins = np.asarray([ray[0] for ray in rays])
    maximum_baseline = 0.0
    for first_index in range(len(origins)):
        for second_index in range(first_index + 1, len(origins)):
            maximum_baseline = max(
                maximum_baseline,
                float(np.linalg.norm(
                    origins[first_index]
                    - origins[second_index]
                )),
            )
    if maximum_baseline < float(minimum_baseline):
        return None

    minimum_cross = math.sin(math.radians(
        float(minimum_parallax_deg)))
    candidates = []
    maximum_parallax = 0.0

    for first_index in range(len(rays)):
        first_origin, first_direction = rays[first_index]
        for second_index in range(
            first_index + 1, len(rays)
        ):
            second_origin, second_direction = rays[second_index]
            denominator = _cross_2d(
                first_direction, second_direction)
            maximum_parallax = max(
                maximum_parallax,
                abs(float(denominator)),
            )
            if abs(denominator) < minimum_cross:
                continue

            delta = second_origin - first_origin
            first_scale = (
                _cross_2d(delta, second_direction)
                / denominator
            )
            second_scale = (
                _cross_2d(delta, first_direction)
                / denominator
            )
            # A physical target must lie in front of both camera bearings.
            if first_scale < -0.10 or second_scale < -0.10:
                continue
            candidates.append(
                first_origin + first_scale * first_direction)

    if not candidates:
        return None

    best_inliers = []
    best_candidate = None
    best_median_error = float('inf')

    for candidate in candidates:
        errors = []
        inliers = []
        for ray_index, (origin, direction) in enumerate(rays):
            distance, along = _point_to_ray_distance(
                candidate, origin, direction)
            if along >= -0.10 and distance <= float(inlier_threshold):
                inliers.append(ray_index)
                errors.append(distance)

        if not inliers:
            continue
        median_error = float(np.median(errors))
        if (
            len(inliers) > len(best_inliers)
            or (
                len(inliers) == len(best_inliers)
                and median_error < best_median_error
            )
        ):
            best_inliers = inliers
            best_candidate = candidate
            best_median_error = median_error

    if best_candidate is None or len(best_inliers) < 3:
        return None

    # Minimize perpendicular distance to all inlier lines:
    #   min_p sum ||n_i^T (p - o_i)||^2
    matrix = np.zeros((2, 2), dtype=np.float64)
    vector = np.zeros(2, dtype=np.float64)
    for ray_index in best_inliers:
        origin, direction = rays[ray_index]
        normal = np.array(
            [-direction[1], direction[0]],
            dtype=np.float64,
        )
        projector = np.outer(normal, normal)
        matrix += projector
        vector += projector @ origin

    if abs(float(np.linalg.det(matrix))) < 1.0e-8:
        return None

    estimate = np.linalg.solve(matrix, vector)
    residuals = []
    valid_inliers = 0
    for ray_index in best_inliers:
        origin, direction = rays[ray_index]
        distance, along = _point_to_ray_distance(
            estimate, origin, direction)
        if along >= -0.10:
            residuals.append(distance)
            valid_inliers += 1

    if valid_inliers < 3:
        return None

    residual = float(np.median(residuals))
    # First-order covariance of the line-intersection estimate.  A small
    # variance floor is applied later during landmark association because the
    # AI-deck calibration is not exact.
    sigma_squared = max(residual, 0.02) ** 2
    covariance = sigma_squared * np.linalg.inv(matrix)

    return {
        'x': float(estimate[0]),
        'y': float(estimate[1]),
        'inliers': int(valid_inliers),
        'residual': residual,
        'baseline': float(maximum_baseline),
        'parallax_sine': float(maximum_parallax),
        'covariance': covariance,
    }



class ArucoDetector:
    def __init__(
        self,
        camera_matrix,
        distortion,
        marker_size_m,
    ):
        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            dictionary, parameters)
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.marker_size_m = float(marker_size_m)

        half = 0.5 * self.marker_size_m
        self.object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )

    def process(
        self,
        frame_bgr,
        drone_pose,
        rotation_body_from_camera,
    ):
        grayscale = cv2.cvtColor(
            frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, identifiers, _rejected = (
            self.detector.detectMarkers(grayscale)
        )
        if identifiers is not None:
            cv2.aruco.drawDetectedMarkers(
                frame_bgr, corners, identifiers)

        results = []
        if identifiers is None or drone_pose is None:
            return results, 0

        for index, corner in enumerate(corners):
            ok, rotation_vector, translation_vector = (
                cv2.solvePnP(
                    self.object_points,
                    corner[0].astype(np.float32),
                    self.camera_matrix,
                    self.distortion,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
            )
            if not ok:
                continue

            point_body = (
                rotation_body_from_camera
                @ translation_vector.flatten()
            )
            point_world = (
                drone_pose.rotation_world_from_body
                @ point_body
            )
            results.append({
                'marker_id': int(
                    identifiers[index][0]),
                'x': drone_pose.x
                + float(point_world[0]),
                'y': drone_pose.y
                + float(point_world[1]),
                'z': drone_pose.z
                + float(point_world[2]),
            })
            cv2.drawFrameAxes(
                frame_bgr,
                self.camera_matrix,
                self.distortion,
                rotation_vector,
                translation_vector,
                0.5 * self.marker_size_m,
            )

        return results, len(results)


class ArucoBackend:
    def __init__(
        self,
        camera_matrix,
        distortion,
        marker_size_m,
        rotation_body_from_camera,
    ):
        self.detector = ArucoDetector(
            camera_matrix,
            distortion,
            marker_size_m,
        )
        self.rotation_body_from_camera = (
            rotation_body_from_camera)

    def process(self, frame_bgr, drone_pose):
        return self.detector.process(
            frame_bgr,
            drone_pose,
            self.rotation_body_from_camera,
        )



def world_point_to_pixel(
    point_world,
    camera_matrix,
    distortion,
    drone_pose,
    rotation_body_from_camera,
):
    """Project a known static world landmark into the current AI-deck image.

    This is the same data-association direction used by feature SLAM systems:
    map landmark -> current image -> bounded search window.  It is much more
    useful for duplicate suppression than comparing two noisy monocular range
    estimates after both have already been triangulated.
    """
    if drone_pose is None:
        return None

    point_world = np.asarray(
        point_world,
        dtype=np.float64,
    ).reshape(3)
    camera_origin = np.asarray(
        [drone_pose.x, drone_pose.y, drone_pose.z],
        dtype=np.float64,
    )
    point_body = (
        drone_pose.rotation_world_from_body.T
        @ (point_world - camera_origin)
    )
    point_camera = (
        rotation_body_from_camera.T
        @ point_body
    )
    depth = float(point_camera[2])
    if depth <= 0.03:
        return {
            'projectable': False,
            'behind': True,
            'u': float('nan'),
            'v': float('nan'),
            'depth': depth,
        }

    projected, _jacobian = cv2.projectPoints(
        point_camera.reshape(1, 1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
    )
    u, v = projected.reshape(-1, 2)[0]
    return {
        'projectable': True,
        'behind': False,
        'u': float(u),
        'v': float(v),
        'depth': depth,
    }



def _nearest_neighbor_cosine_distance(
    gallery,
    feature,
):
    """DeepSORT-style minimum cosine distance to an identity gallery."""
    if feature is None or not gallery:
        return float('inf')
    query = np.asarray(
        feature,
        dtype=np.float32,
    ).reshape(-1)
    query_norm = float(np.linalg.norm(query))
    if query_norm <= 1.0e-8:
        return float('inf')
    query = query / query_norm

    samples = np.asarray(
        list(gallery),
        dtype=np.float32,
    )
    if samples.ndim != 2 or samples.shape[1] != query.shape[0]:
        return float('inf')
    norms = np.linalg.norm(
        samples,
        axis=1,
        keepdims=True,
    )
    samples = samples / np.maximum(
        norms,
        1.0e-8,
    )
    distances = 1.0 - samples @ query
    return float(np.min(distances))


def _append_normalized_feature(
    gallery,
    feature,
):
    if feature is None:
        return
    descriptor = np.asarray(
        feature,
        dtype=np.float32,
    ).reshape(-1)
    norm = float(np.linalg.norm(descriptor))
    if norm <= 1.0e-8:
        return
    gallery.append(
        descriptor / norm)


class _FastTracklet:

    """Small fixed-window ray bundle for one local ByteTrack identity."""

    def __init__(
        self,
        source_id,
        local_track_id,
        class_id,
        class_name,
        sample_window,
    ):
        self.source_id = str(source_id)
        self.local_track_id = int(local_track_id)
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        self.samples = deque(
            maxlen=max(6, int(sample_window)))
        self.first_seen = 0.0
        self.last_seen = 0.0
        self.last_bbox = None
        self.last_image_center = (0.0, 0.0)
        self.appearance_bank = deque(maxlen=16)
        self.unexplained_count = 0

    @property
    def key(self):
        return (
            self.source_id,
            self.local_track_id,
        )

    def update(self, detection, now):
        if not self.samples:
            self.first_seen = float(now)
        self.last_seen = float(now)
        self.last_bbox = tuple(
            float(value)
            for value in detection['bbox']
        )
        self.last_image_center = tuple(
            float(value)
            for value in detection['image_center']
        )
        _append_normalized_feature(
            self.appearance_bank,
            detection.get('appearance'),
        )
        self.samples.append((
            float(detection['x']),
            float(detection['y']),
            float(detection['z']),
            float(detection['confidence']),
            float(detection['ray_origin_x']),
            float(detection['ray_origin_y']),
            float(detection['ray_direction_x']),
            float(detection['ray_direction_y']),
            float(now),
        ))

    def ground_center(self):
        if not self.samples:
            return None
        points = np.asarray(
            [
                (sample[0], sample[1])
                for sample in self.samples
            ],
            dtype=np.float64,
        )
        return np.median(points, axis=0)

    def estimate(
        self,
        inlier_threshold,
        minimum_baseline,
    ):
        return _incremental_ray_fit(
            self.samples,
            inlier_threshold=inlier_threshold,
            minimum_baseline=minimum_baseline,
        )

    def ray_error(self, point, recent=8):
        samples = list(self.samples)[-max(1, int(recent)):]
        errors = []
        for sample in samples:
            distance, along = _point_to_ray_distance(
                point,
                (sample[4], sample[5]),
                (sample[6], sample[7]),
            )
            if along >= -0.10:
                errors.append(float(distance))
        if not errors:
            return float('inf')
        return float(np.median(errors))

    def appearance_signature(self):
        if not self.appearance_bank:
            return None
        features = np.asarray(
            list(self.appearance_bank),
            dtype=np.float32,
        )
        signature = np.median(
            features,
            axis=0,
        ).astype(np.float32)
        norm = float(np.linalg.norm(signature))
        if norm <= 1.0e-8:
            return None
        return signature / norm

    def appearance_distance_to(self, gallery):
        return _nearest_neighbor_cosine_distance(
            gallery,
            self.appearance_signature(),
        )


class _FastLandmark:
    """Published landmark with a non-drifting identity anchor.

    A bad fragmented track must never drag the landmark centre far enough that
    the next revisit becomes a second marker.  The landmark therefore keeps a
    small bank of independently fitted anchor points.  Only geometrically gated
    tracklets may update that bank or the ray bundle.
    """

    def __init__(
        self,
        marker_id,
        tracklet,
        sample_window,
        initial_fit=None,
    ):
        self.marker_id = int(marker_id)
        self.source_id = tracklet.source_id
        self.class_id = tracklet.class_id
        self.class_name = tracklet.class_name
        self.track_keys = set()
        self.samples = deque(
            maxlen=max(16, int(sample_window)))
        self.anchor_points = deque(maxlen=7)
        self.appearance_bank = deque(maxlen=48)
        self.last_seen = tracklet.last_seen
        self.absorb_tracklet(
            tracklet,
            fit=initial_fit,
            update_geometry=True,
        )

    def anchor_center(self):
        if not self.anchor_points:
            return None
        points = np.asarray(
            self.anchor_points,
            dtype=np.float64,
        )
        return np.median(points, axis=0)

    def absorb_tracklet(
        self,
        tracklet,
        fit=None,
        update_geometry=True,
    ):
        is_new = tracklet.key not in self.track_keys
        self.track_keys.add(tracklet.key)
        self.last_seen = max(
            self.last_seen,
            tracklet.last_seen,
        )

        # DeepSORT-style identity gallery is updated independently of the
        # geometric anchor.  Geometry may be rejected while appearance still
        # helps reconnect the next fragmented local ID.
        for feature in tracklet.appearance_bank:
            _append_normalized_feature(
                self.appearance_bank,
                feature,
            )

        if not update_geometry:
            return

        # Never let a newly associated fragmented track drag the map point.
        # Identity and geometry updates are deliberately separate decisions.
        anchor = self.anchor_center()
        if fit is None:
            update_geometry = not is_new
        elif anchor is not None:
            candidate = np.asarray(
                [fit['x'], fit['y']],
                dtype=np.float64,
            )
            covariance = np.asarray(
                fit.get('covariance', np.eye(2) * 0.04),
                dtype=np.float64,
            )
            delta = candidate - anchor
            try:
                normalized_error = float(
                    delta.T
                    @ np.linalg.inv(
                        covariance + np.eye(2) * 0.04 ** 2
                    )
                    @ delta
                )
            except np.linalg.LinAlgError:
                normalized_error = float('inf')
            update_geometry = normalized_error <= 9.210

        if not update_geometry:
            return

        if is_new:
            for sample in tracklet.samples:
                self.samples.append(sample)
        elif tracklet.samples:
            self.samples.append(tracklet.samples[-1])

        if fit is not None:
            candidate = np.asarray(
                [fit['x'], fit['y']],
                dtype=np.float64,
            )
            anchor = self.anchor_center()
            if anchor is None or float(np.linalg.norm(candidate - anchor)) <= 0.40:
                self.anchor_points.append(candidate)

    def appearance_distance(self, tracklet_or_candidate):
        return tracklet_or_candidate.appearance_distance_to(
            self.appearance_bank)

    def estimate(
        self,
        inlier_threshold,
        minimum_baseline,
    ):
        return _incremental_ray_fit(
            self.samples,
            inlier_threshold=inlier_threshold,
            minimum_baseline=minimum_baseline,
        )

    def center(
        self,
        inlier_threshold,
        minimum_baseline,
    ):
        # Identity association uses the robust anchor, not a ray fit that can
        # be dragged by one wrongly linked tracklet.
        anchor = self.anchor_center()
        if anchor is not None:
            return np.asarray(
                anchor,
                dtype=np.float64,
            )

        fit = self.estimate(
            inlier_threshold,
            minimum_baseline,
        )
        if fit is not None:
            return np.asarray(
                [fit['x'], fit['y']],
                dtype=np.float64,
            )

        points = np.asarray(
            [
                (sample[0], sample[1])
                for sample in self.samples
            ],
            dtype=np.float64,
        )
        return np.median(points, axis=0)


def _incremental_ray_fit(
    samples,
    inlier_threshold,
    minimum_baseline,
):
    """Robust ground-plane point fusion with covariance.

    Every AI-deck detection has already been intersected with the known ground
    plane, so each sample is a complete 2-D position measurement.  Treating the
    rays as a free-space triangulation problem makes a straight fly-by look
    ill-conditioned even though the ground constraint already solved depth.

    The estimator therefore fuses the direct ground hits with Huber weights,
    uses camera baseline only as an independence check, and returns a robust
    covariance for Mahalanobis association.
    """
    samples = list(samples)
    if len(samples) < 3:
        return None

    points = np.asarray(
        [(sample[0], sample[1]) for sample in samples],
        dtype=np.float64,
    )
    origins = np.asarray(
        [(sample[4], sample[5]) for sample in samples],
        dtype=np.float64,
    )
    confidences = np.clip(
        np.asarray([sample[3] for sample in samples], dtype=np.float64),
        0.10,
        1.0,
    )
    finite = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(origins).all(axis=1)
        & np.isfinite(confidences)
    )
    points = points[finite]
    origins = origins[finite]
    confidences = confidences[finite]
    if len(points) < 3:
        return None

    delta_origin = origins[:, None, :] - origins[None, :, :]
    maximum_baseline = float(
        np.max(np.linalg.norm(delta_origin, axis=2))
    )
    if maximum_baseline < float(minimum_baseline):
        return None

    median = np.median(points, axis=0)
    radial = np.linalg.norm(points - median[None, :], axis=1)
    radial_median = float(np.median(radial))
    radial_mad = float(np.median(np.abs(radial - radial_median)))
    robust_scale = max(0.015, 1.4826 * radial_mad, radial_median)
    adaptive_limit = max(
        float(inlier_threshold),
        2.8 * robust_scale,
    )
    inlier_mask = radial <= adaptive_limit
    if np.count_nonzero(inlier_mask) < 3:
        # Keep the three observations nearest the robust centre instead of
        # failing a brief but otherwise coherent fly-by.
        nearest = np.argsort(radial)[:3]
        inlier_mask = np.zeros(len(points), dtype=bool)
        inlier_mask[nearest] = True

    selected_points = points[inlier_mask]
    selected_confidences = confidences[inlier_mask]
    selected_radial = np.linalg.norm(
        selected_points - median[None, :],
        axis=1,
    )
    normalized = selected_radial / max(adaptive_limit, 1.0e-6)
    huber = np.where(
        normalized <= 0.5,
        1.0,
        np.maximum(0.20, 0.5 / np.maximum(normalized, 1.0e-6)),
    )
    weights = selected_confidences * huber
    weight_sum = max(float(np.sum(weights)), 1.0e-8)
    refined = np.sum(
        selected_points * weights[:, None],
        axis=0,
    ) / weight_sum

    errors = np.linalg.norm(
        selected_points - refined[None, :],
        axis=1,
    )
    residual = float(np.median(errors))

    centered = selected_points - refined[None, :]
    scatter = np.zeros((2, 2), dtype=np.float64)
    for vector, weight in zip(centered, weights):
        scatter += float(weight) * np.outer(vector, vector)
    effective_count = (
        weight_sum * weight_sum
        / max(float(np.sum(weights * weights)), 1.0e-8)
    )
    scatter /= max(weight_sum, 1.0e-8)
    covariance = scatter / max(1.0, effective_count)

    ranges = np.linalg.norm(
        refined[None, :] - origins[inlier_mask],
        axis=1,
    )
    median_range = float(np.median(ranges)) if len(ranges) else 0.0
    # This floor models pixel quantization, pose interpolation and imperfect
    # calibration.  It scales with range automatically rather than being tuned
    # per arena.
    systematic_sigma = 0.020 + 0.030 * min(4.0, median_range)
    covariance += np.eye(2, dtype=np.float64) * systematic_sigma ** 2
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.025 ** 2, 0.60 ** 2)
    covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    radius95 = float(math.sqrt(5.991 * float(np.max(eigenvalues))))
    condition_number = float(
        max(eigenvalues) / max(min(eigenvalues), 1.0e-12)
    )

    return {
        'x': float(refined[0]),
        'y': float(refined[1]),
        'inliers': int(np.count_nonzero(inlier_mask)),
        'residual': residual,
        'baseline': maximum_baseline,
        'covariance': covariance,
        'radius95': radius95,
        'condition_number': condition_number,
        'median_range': median_range,
    }

def _group_fit_stats(
    samples,
    point,
    threshold,
):
    errors = []
    inliers = 0
    for sample in samples:
        distance, along = _point_to_ray_distance(
            point,
            (sample[4], sample[5]),
            (sample[6], sample[7]),
        )
        if along < -0.10:
            continue
        errors.append(float(distance))
        if distance <= float(threshold):
            inliers += 1

    if not errors:
        return {
            'valid': 0,
            'ratio': 0.0,
            'median': float('inf'),
        }
    return {
        'valid': len(errors),
        'ratio': (
            float(inliers)
            / float(len(errors))
        ),
        'median': float(np.median(errors)),
    }



class _DeferredLandmarkCandidate:
    """AFLink-style deferred graph component for unpublished tracklets."""

    def __init__(
        self,
        candidate_id,
        tracklet,
        sample_window,
    ):
        self.candidate_id = int(candidate_id)
        self.source_id = tracklet.source_id
        self.class_id = tracklet.class_id
        self.class_name = tracklet.class_name
        self.track_keys = set()
        self.samples = deque(
            maxlen=max(16, int(sample_window)))
        self.appearance_bank = deque(maxlen=48)
        self.first_seen = float(tracklet.first_seen)
        self.last_seen = float(tracklet.last_seen)
        self.distinct_evidence = {}
        self.hold_logged = False
        self.absorb(tracklet)

    def absorb(self, tracklet):
        is_new = tracklet.key not in self.track_keys
        self.track_keys.add(tracklet.key)
        self.first_seen = min(
            self.first_seen,
            float(tracklet.first_seen),
        )
        self.last_seen = max(
            self.last_seen,
            float(tracklet.last_seen),
        )
        for feature in tracklet.appearance_bank:
            _append_normalized_feature(
                self.appearance_bank,
                feature,
            )
        if is_new:
            for sample in tracklet.samples:
                self.samples.append(sample)
        elif tracklet.samples:
            self.samples.append(
                tracklet.samples[-1])

    def appearance_signature(self):
        if not self.appearance_bank:
            return None
        features = np.asarray(
            list(self.appearance_bank),
            dtype=np.float32,
        )
        signature = np.median(
            features,
            axis=0,
        ).astype(np.float32)
        norm = float(np.linalg.norm(signature))
        if norm <= 1.0e-8:
            return None
        return signature / norm

    def appearance_distance_to(self, gallery):
        return _nearest_neighbor_cosine_distance(
            gallery,
            self.appearance_signature(),
        )

    def estimate(
        self,
        inlier_threshold,
        minimum_baseline,
    ):
        return _incremental_ray_fit(
            self.samples,
            inlier_threshold=inlier_threshold,
            minimum_baseline=minimum_baseline,
        )

    def ground_center(self):
        if not self.samples:
            return None
        points = np.asarray(
            [
                (sample[0], sample[1])
                for sample in self.samples
            ],
            dtype=np.float64,
        )
        return np.median(points, axis=0)

    def ray_error(self, point, recent=12):
        samples = list(self.samples)[-max(1, int(recent)):]
        errors = []
        for sample in samples:
            distance, along = _point_to_ray_distance(
                point,
                (sample[4], sample[5]),
                (sample[6], sample[7]),
            )
            if along >= -0.10:
                errors.append(float(distance))
        if not errors:
            return float('inf')
        return float(np.median(errors))


class RealtimeLandmarkRegistry:
    """Projection-first deferred landmark graph.

    The v12 inference path stays untouched.  Only landmark identity changes.

    The registry combines three proven patterns:
    - ORB-SLAM-style map-point projection into the current image before a new
      landmark is created;
    - UCMCTrack-style one-to-one cost-matrix assignment instead of greedy
      per-track matching;
    - StrongSORT AFLink-style deferred linking of fragmented local tracklets.

    Every unmatched local ByteTrack ID first joins an unpublished candidate.
    A candidate is published only after existing landmarks have repeatedly
    failed to explain it.  This makes duplicate suppression conservative while
    still allowing a true sequential object to publish after a few frames.
    """

    FIRST_MARKER_ID = 1_000_000
    INVALID_COST = 1.0e6

    def __init__(
        self,
        logger,
        sample_window=24,
        confirmation_hits=3,
        minimum_confirmation_age=0.4,
        minimum_triangulation_inliers=3,
        maximum_ray_residual=0.24,
        ray_inlier_threshold=0.24,
        ray_match_threshold=0.20,
        minimum_parallax_deg=1.0,
        minimum_baseline=0.06,
        covariance_floor=0.20,
        association_chi2_gate=16.0,
        association_max_distance=0.60,
        ambiguity_chi2_gate=36.0,
        ambiguity_max_distance=1.20,
        duplicate_minimum_bbox_iou=0.15,
        duplicate_maximum_pixel_distance=24.0,
        distinct_evidence_frames=2,
        distinct_maximum_bbox_iou=0.15,
        distinct_minimum_pixel_distance=20.0,
        bundle_inlier_threshold=0.18,
        bundle_minimum_group_inlier_ratio=0.75,
        bundle_maximum_group_median_error=0.18,
        appearance_merge_threshold=0.12,
        appearance_max_distance=0.90,
        hypothesis_spatial_gate=0.45,
        hypothesis_minimum_tracklets=2,
        hypothesis_minimum_separation=0.45,
        hypothesis_separation_chi2=12.0,
        hypothesis_timeout=60.0,
        tracklet_timeout=18.0,
    ):
        del (
            minimum_parallax_deg,
            covariance_floor,
            association_chi2_gate,
            ambiguity_chi2_gate,
            hypothesis_minimum_tracklets,
            hypothesis_separation_chi2,
        )

        self.logger = logger
        self.sample_window = max(
            8, min(32, int(sample_window)))
        self.confirmation_hits = max(
            3, int(confirmation_hits))
        self.minimum_confirmation_age = max(
            0.20,
            float(minimum_confirmation_age),
        )
        self.minimum_triangulation_inliers = max(
            3,
            int(minimum_triangulation_inliers),
        )
        self.maximum_ray_residual = max(
            0.05,
            float(maximum_ray_residual),
        )
        self.ray_inlier_threshold = max(
            self.maximum_ray_residual,
            float(ray_inlier_threshold),
        )
        self.ray_match_threshold = max(
            0.12,
            float(ray_match_threshold),
        )
        self.minimum_baseline = max(
            0.04,
            float(minimum_baseline),
        )

        # Strong metric merge remains below 0.4 m because the mission may place
        # two real targets around 0.6 m apart.
        self.direct_merge_distance = min(
            0.38,
            max(
                0.22,
                float(association_max_distance),
            ),
        )
        # This is only an ambiguity veto, not an automatic merge radius.
        self.ambiguity_max_distance = max(
            0.75,
            min(
                1.20,
                float(ambiguity_max_distance),
            ),
        )
        self.new_object_minimum_separation = min(
            0.40,
            max(
                0.28,
                float(hypothesis_minimum_separation),
            ),
        )
        self.candidate_spatial_gate = min(
            0.50,
            max(
                0.32,
                float(hypothesis_spatial_gate),
            ),
        )

        # DeepSORT's gallery uses the nearest descriptor observed for a target
        # rather than one averaged template.  Keep the YAML threshold but clamp
        # it to the range that is useful for the lightweight descriptor.
        self.appearance_match_threshold = min(
            0.30,
            max(
                0.14,
                float(appearance_merge_threshold),
            ),
        )
        self.appearance_ambiguity_threshold = min(
            0.42,
            max(
                0.26,
                self.appearance_match_threshold + 0.10,
            ),
        )
        self.appearance_max_distance = max(
            self.direct_merge_distance,
            float(appearance_max_distance),
        )

        self.duplicate_minimum_bbox_iou = min(
            1.0,
            max(
                0.05,
                float(duplicate_minimum_bbox_iou),
            ),
        )
        self.duplicate_maximum_pixel_distance = max(
            18.0,
            float(duplicate_maximum_pixel_distance),
        )
        self.distinct_evidence_frames = max(
            2,
            int(distinct_evidence_frames),
        )
        self.distinct_maximum_bbox_iou = min(
            0.30,
            max(
                0.0,
                float(distinct_maximum_bbox_iou),
            ),
        )
        self.distinct_minimum_pixel_distance = max(
            14.0,
            float(distinct_minimum_pixel_distance),
        )

        self.bundle_inlier_threshold = max(
            self.ray_inlier_threshold,
            float(bundle_inlier_threshold),
        )
        self.bundle_minimum_group_inlier_ratio = min(
            1.0,
            max(
                0.55,
                float(bundle_minimum_group_inlier_ratio),
            ),
        )
        self.bundle_maximum_group_median_error = max(
            0.12,
            float(bundle_maximum_group_median_error),
        )

        self.tracklet_timeout = max(
            8.0,
            float(tracklet_timeout),
        )
        self.candidate_timeout = max(
            12.0,
            min(
                45.0,
                float(hypothesis_timeout),
            ),
        )

        self._tracklets = {}
        self._landmarks = {}
        self._candidates = {}
        self._track_to_marker = {}
        self._track_to_candidate = {}
        self._next_marker_id = self.FIRST_MARKER_ID
        self._next_candidate_id = 1
        self._lock = threading.Lock()

    @staticmethod
    def _distance(first, second):
        return math.hypot(
            float(first[0]) - float(second[0]),
            float(first[1]) - float(second[1]),
        )

    @staticmethod
    def _bbox_iou(first, second):
        if first is None or second is None:
            return 0.0
        first_x, first_y, first_w, first_h = first
        second_x, second_y, second_w, second_h = second
        first_x2 = first_x + first_w
        first_y2 = first_y + first_h
        second_x2 = second_x + second_w
        second_y2 = second_y + second_h

        width = max(
            0.0,
            min(first_x2, second_x2)
            - max(first_x, second_x),
        )
        height = max(
            0.0,
            min(first_y2, second_y2)
            - max(first_y, second_y),
        )
        intersection = width * height
        union = (
            first_w * first_h
            + second_w * second_h
            - intersection
        )
        if union <= 1.0e-9:
            return 0.0
        return float(intersection / union)

    @staticmethod
    def _hungarian(cost_matrix, unmatched_cost):
        """Small dependency-free Hungarian assignment."""
        costs = np.asarray(
            cost_matrix,
            dtype=np.float64,
        )
        if costs.ndim != 2:
            return []
        row_count, real_column_count = costs.shape
        if row_count == 0 or real_column_count == 0:
            return []

        padded = np.full(
            (
                row_count,
                real_column_count + row_count,
            ),
            float(unmatched_cost),
            dtype=np.float64,
        )
        padded[:, :real_column_count] = costs

        n, m = padded.shape
        u = np.zeros(n + 1, dtype=np.float64)
        v = np.zeros(m + 1, dtype=np.float64)
        p = np.zeros(m + 1, dtype=np.int64)
        way = np.zeros(m + 1, dtype=np.int64)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = np.full(
                m + 1,
                np.inf,
                dtype=np.float64,
            )
            used = np.zeros(
                m + 1,
                dtype=bool,
            )

            while True:
                used[j0] = True
                i0 = int(p[j0])
                delta = np.inf
                j1 = 0
                for j in range(1, m + 1):
                    if used[j]:
                        continue
                    current = (
                        padded[i0 - 1, j - 1]
                        - u[i0]
                        - v[j]
                    )
                    if current < minv[j]:
                        minv[j] = current
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

                for j in range(m + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break

            while True:
                j1 = int(way[j0])
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        assignment = [-1] * n
        for j in range(1, m + 1):
            if p[j] != 0:
                assignment[int(p[j]) - 1] = j - 1

        output = []
        for row_index, column_index in enumerate(
            assignment
        ):
            if (
                0 <= column_index < real_column_count
                and padded[row_index, column_index]
                < float(unmatched_cost)
            ):
                output.append(
                    (row_index, column_index))
        return output

    def _landmarks_for(
        self,
        source_id,
        class_id,
    ):
        return [
            landmark
            for landmark in self._landmarks.values()
            if (
                landmark.source_id == str(source_id)
                and landmark.class_id == int(class_id)
            )
        ]

    def _candidates_for(
        self,
        source_id,
        class_id,
    ):
        return [
            candidate
            for candidate in self._candidates.values()
            if (
                candidate.source_id == str(source_id)
                and candidate.class_id == int(class_id)
            )
        ]

    def _active_tracklets(
        self,
        track_keys,
        current_keys,
    ):
        return [
            self._tracklets[key]
            for key in track_keys
            if (
                key in current_keys
                and key in self._tracklets
            )
        ]

    def _distinct_pairs(self, tracklets):
        pairs = set()
        for first_index in range(len(tracklets)):
            first = tracklets[first_index]
            for second_index in range(
                first_index + 1,
                len(tracklets),
            ):
                second = tracklets[second_index]
                if (
                    first.source_id != second.source_id
                    or first.class_id != second.class_id
                ):
                    continue
                iou = self._bbox_iou(
                    first.last_bbox,
                    second.last_bbox,
                )
                pixel_distance = self._distance(
                    first.last_image_center,
                    second.last_image_center,
                )
                if (
                    iou
                    <= self.distinct_maximum_bbox_iou
                    and pixel_distance
                    >= self.distinct_minimum_pixel_distance
                ):
                    pairs.add(frozenset(
                        (first.key, second.key)))
        return pairs

    def _has_cannot_link(
        self,
        first_keys,
        second_keys,
        cannot_link_pairs,
    ):
        for first_key in first_keys:
            for second_key in second_keys:
                if frozenset(
                    (first_key, second_key)
                ) in cannot_link_pairs:
                    return True
        return False

    def _combined_samples_compatible(
        self,
        first_samples,
        second_samples,
    ):
        first_samples = list(first_samples)[-16:]
        second_samples = list(second_samples)[-16:]
        if (
            len(first_samples) < 3
            or len(second_samples) < 3
        ):
            return False, None

        fit = _incremental_ray_fit(
            first_samples + second_samples,
            inlier_threshold=(
                self.bundle_inlier_threshold),
            minimum_baseline=(
                self.minimum_baseline),
        )
        if fit is None:
            return False, None

        point = (fit['x'], fit['y'])
        first_stats = _group_fit_stats(
            first_samples,
            point,
            self.bundle_inlier_threshold,
        )
        second_stats = _group_fit_stats(
            second_samples,
            point,
            self.bundle_inlier_threshold,
        )
        compatible = (
            first_stats['valid'] >= 3
            and second_stats['valid'] >= 3
            and first_stats['ratio']
            >= self.bundle_minimum_group_inlier_ratio
            and second_stats['ratio']
            >= self.bundle_minimum_group_inlier_ratio
            and first_stats['median']
            <= self.bundle_maximum_group_median_error
            and second_stats['median']
            <= self.bundle_maximum_group_median_error
        )
        return bool(compatible), {
            'fit': fit,
            'first': first_stats,
            'second': second_stats,
        }

    def _projection_relation(
        self,
        tracklet,
        landmark,
        projection_context,
    ):
        output = {
            'available': False,
            'projectable': False,
            'inside_margin': False,
            'inside_image': False,
            'pixel_distance': float('inf'),
            'strong_radius': float('inf'),
            'ambiguous_radius': float('inf'),
            'far': False,
            'u': float('nan'),
            'v': float('nan'),
        }
        if projection_context is None:
            return output

        landmark_center = landmark.center(
            self.ray_inlier_threshold,
            self.minimum_baseline,
        )
        projected = world_point_to_pixel(
            (
                float(landmark_center[0]),
                float(landmark_center[1]),
                float(projection_context['ground_z']),
            ),
            projection_context['camera_matrix'],
            projection_context['distortion'],
            projection_context['drone_pose'],
            projection_context[
                'rotation_body_from_camera'],
        )
        output['available'] = True
        if (
            projected is None
            or not projected.get(
                'projectable', False)
        ):
            output['far'] = True
            return output

        u = float(projected['u'])
        v = float(projected['v'])
        output['projectable'] = True
        output['u'] = u
        output['v'] = v

        image_height = float(
            projection_context['image_height'])
        image_width = float(
            projection_context['image_width'])
        x, y, width, height = tracklet.last_bbox
        diagonal = math.hypot(width, height)
        maximum_side = max(width, height)
        strong_radius = max(
            18.0,
            0.65 * maximum_side,
        )
        ambiguous_radius = max(
            38.0,
            1.20 * maximum_side,
        )
        center_u, center_v = (
            tracklet.last_image_center)
        pixel_distance = math.hypot(
            u - center_u,
            v - center_v,
        )
        output['pixel_distance'] = pixel_distance
        output['strong_radius'] = strong_radius
        output['ambiguous_radius'] = ambiguous_radius
        output['inside_image'] = (
            0.0 <= u < image_width
            and 0.0 <= v < image_height
        )
        output['inside_margin'] = (
            -ambiguous_radius <= u
            < image_width + ambiguous_radius
            and -ambiguous_radius <= v
            < image_height + ambiguous_radius
        )
        output['far'] = (
            not output['inside_margin']
            or pixel_distance
            > ambiguous_radius
        )
        return output

    def _tracklet_landmark_relation(
        self,
        tracklet,
        landmark,
        projection_context,
        current_keys,
        cannot_link_pairs,
    ):
        active_landmark = self._active_tracklets(
            landmark.track_keys,
            current_keys,
        )
        if self._has_cannot_link(
            {tracklet.key},
            {item.key for item in active_landmark},
            cannot_link_pairs,
        ):
            return {
                'strong': False,
                'ambiguous': False,
                'distinct': True,
                'cannot_link': True,
                'score': self.INVALID_COST,
                'reason': 'cannot-link',
                'geometry_update_ok': False,
                'appearance_distance': float('inf'),
            }

        landmark_center = landmark.center(
            self.ray_inlier_threshold,
            self.minimum_baseline,
        )
        track_fit = tracklet.estimate(
            self.ray_inlier_threshold,
            self.minimum_baseline,
        )
        ground_center = tracklet.ground_center()
        fit_distance = (
            self._distance(
                (track_fit['x'], track_fit['y']),
                landmark_center,
            )
            if track_fit is not None
            else float('inf')
        )
        ground_distance = (
            self._distance(
                ground_center,
                landmark_center,
            )
            if ground_center is not None
            else float('inf')
        )
        ray_error = tracklet.ray_error(
            landmark_center,
            recent=10,
        )
        appearance_distance = landmark.appearance_distance(
            tracklet)
        bundle_ok, bundle = (
            self._combined_samples_compatible(
                tracklet.samples,
                landmark.samples,
            )
        )
        projection = self._projection_relation(
            tracklet,
            landmark,
            projection_context,
        )

        projection_strong = (
            projection['projectable']
            and projection['pixel_distance']
            <= projection['strong_radius']
            and (
                ray_error <= 0.48
                or fit_distance <= 0.70
                or ground_distance <= 0.70
            )
        )
        geometry_strong = (
            fit_distance <= 0.32
            and ray_error <= 0.34
        )
        bearing_strong = (
            ray_error <= self.ray_match_threshold
            and ground_distance
            <= self.ambiguity_max_distance
        )
        bundle_strong = (
            bundle_ok
            and ray_error <= 0.28
        )
        appearance_strong = (
            appearance_distance
            <= self.appearance_match_threshold
        )
        projection_ambiguous = (
            projection['projectable']
            and projection['pixel_distance']
            <= projection['ambiguous_radius']
        )
        appearance_guided = (
            appearance_strong
            and (
                projection_ambiguous
                or fit_distance <= 0.62
                or ground_distance <= 0.62
                or ray_error <= 0.38
            )
        )
        strong = (
            projection_strong
            or geometry_strong
            or bearing_strong
            or bundle_strong
            or appearance_guided
        )

        appearance_ambiguous = (
            appearance_distance
            <= self.appearance_ambiguity_threshold
        )
        geometry_decisive = (
            fit_distance >= 0.58
            and ray_error >= 0.40
            and not bundle_ok
        )

        if projection['available']:
            ambiguous = (
                not strong
                and (
                    projection_ambiguous
                    or bundle_ok
                    or ray_error <= 0.42
                    or (
                        appearance_ambiguous
                        and not geometry_decisive
                    )
                )
            )
            distinct = (
                not strong
                and not ambiguous
                and projection['far']
                and fit_distance >= 0.38
                and ray_error >= 0.28
                and not bundle_ok
                and (
                    not appearance_ambiguous
                    or geometry_decisive
                )
            )
        else:
            ambiguous = (
                not strong
                and (
                    fit_distance <= 0.78
                    or ground_distance <= 0.78
                    or ray_error <= 0.58
                    or bundle_ok
                    or (
                        appearance_ambiguous
                        and not geometry_decisive
                    )
                )
            )
            distinct = (
                not strong
                and not ambiguous
                and fit_distance >= 0.48
                and ray_error >= 0.40
                and not bundle_ok
                and (
                    not appearance_ambiguous
                    or (
                        fit_distance >= 0.68
                        and ray_error >= 0.46
                    )
                )
            )

        score = (
            min(
                projection['pixel_distance']
                / max(
                    projection['strong_radius'],
                    1.0,
                ),
                4.0,
            )
            if projection['projectable']
            else 2.0
        )
        score += 0.80 * min(
            fit_distance
            / max(
                self.direct_merge_distance,
                1.0e-6,
            ),
            4.0,
        )
        score += 0.60 * min(
            ray_error
            / max(
                self.ray_match_threshold,
                1.0e-6,
            ),
            4.0,
        )
        if math.isfinite(appearance_distance):
            score += 0.90 * min(
                appearance_distance
                / max(
                    self.appearance_match_threshold,
                    1.0e-6,
                ),
                4.0,
            )
        if bundle_strong:
            score -= 1.0
        if appearance_guided:
            score -= 0.75

        geometry_update_ok = (
            fit_distance <= 0.30
            or (
                bundle_strong
                and ray_error <= 0.18
            )
        )
        reason = (
            'projection'
            if projection_strong
            else (
                'position'
                if geometry_strong
                else (
                    'bearing'
                    if bearing_strong
                    else (
                        'bundle'
                        if bundle_strong
                        else 'appearance'
                    )
                )
            )
        )
        return {
            'strong': bool(strong),
            'ambiguous': bool(ambiguous),
            'distinct': bool(distinct),
            'cannot_link': False,
            'score': float(score),
            'reason': reason,
            'fit_distance': fit_distance,
            'ground_distance': ground_distance,
            'ray_error': ray_error,
            'appearance_distance': appearance_distance,
            'bundle': bundle,
            'projection': projection,
            'geometry_update_ok': bool(
                geometry_update_ok),
        }

    def _candidate_landmark_relation(
        self,
        candidate,
        active_tracklet,
        landmark,
        projection_context,
        current_keys,
        cannot_link_pairs,
    ):
        active_landmark = self._active_tracklets(
            landmark.track_keys,
            current_keys,
        )
        active_candidate = self._active_tracklets(
            candidate.track_keys,
            current_keys,
        )
        cannot_link = self._has_cannot_link(
            {item.key for item in active_candidate},
            {item.key for item in active_landmark},
            cannot_link_pairs,
        )
        if cannot_link:
            return {
                'strong': False,
                'ambiguous': False,
                'distinct': True,
                'cannot_link': True,
                'score': self.INVALID_COST,
                'reason': 'cannot-link',
                'geometry_update_ok': False,
                'appearance_distance': float('inf'),
            }

        landmark_center = landmark.center(
            self.ray_inlier_threshold,
            self.minimum_baseline,
        )
        candidate_fit = candidate.estimate(
            self.ray_inlier_threshold,
            self.minimum_baseline,
        )
        ground_center = candidate.ground_center()
        fit_distance = (
            self._distance(
                (
                    candidate_fit['x'],
                    candidate_fit['y'],
                ),
                landmark_center,
            )
            if candidate_fit is not None
            else float('inf')
        )
        ground_distance = (
            self._distance(
                ground_center,
                landmark_center,
            )
            if ground_center is not None
            else float('inf')
        )
        ray_error = candidate.ray_error(
            landmark_center,
            recent=14,
        )
        appearance_distance = (
            candidate.appearance_distance_to(
                landmark.appearance_bank)
        )
        bundle_ok, bundle = (
            self._combined_samples_compatible(
                candidate.samples,
                landmark.samples,
            )
        )
        projection = self._projection_relation(
            active_tracklet,
            landmark,
            projection_context,
        )

        projection_strong = (
            projection['projectable']
            and projection['pixel_distance']
            <= projection['strong_radius']
            and (
                ray_error <= 0.50
                or fit_distance <= 0.72
                or ground_distance <= 0.72
            )
        )
        geometry_strong = (
            fit_distance <= 0.34
            and ray_error <= 0.36
        )
        bearing_strong = (
            ray_error <= self.ray_match_threshold
            and ground_distance
            <= self.ambiguity_max_distance
        )
        bundle_strong = (
            bundle_ok
            and ray_error <= 0.30
        )
        appearance_strong = (
            appearance_distance
            <= self.appearance_match_threshold
        )
        projection_ambiguous = (
            projection['projectable']
            and projection['pixel_distance']
            <= projection['ambiguous_radius']
        )
        appearance_guided = (
            appearance_strong
            and (
                projection_ambiguous
                or fit_distance <= 0.66
                or ground_distance <= 0.66
                or ray_error <= 0.40
            )
        )
        strong = (
            projection_strong
            or geometry_strong
            or bearing_strong
            or bundle_strong
            or appearance_guided
        )

        appearance_ambiguous = (
            appearance_distance
            <= self.appearance_ambiguity_threshold
        )
        geometry_decisive = (
            fit_distance >= 0.60
            and ray_error >= 0.42
            and not bundle_ok
        )

        if projection['available']:
            ambiguous = (
                not strong
                and (
                    projection_ambiguous
                    or bundle_ok
                    or ray_error <= 0.44
                    or (
                        appearance_ambiguous
                        and not geometry_decisive
                    )
                )
            )
            distinct = (
                not strong
                and not ambiguous
                and projection['far']
                and fit_distance >= 0.38
                and ray_error >= 0.30
                and not bundle_ok
                and (
                    not appearance_ambiguous
                    or geometry_decisive
                )
            )
        else:
            ambiguous = (
                not strong
                and (
                    fit_distance <= 0.82
                    or ground_distance <= 0.82
                    or ray_error <= 0.62
                    or bundle_ok
                    or (
                        appearance_ambiguous
                        and not geometry_decisive
                    )
                )
            )
            distinct = (
                not strong
                and not ambiguous
                and fit_distance >= 0.50
                and ray_error >= 0.42
                and not bundle_ok
                and (
                    not appearance_ambiguous
                    or (
                        fit_distance >= 0.70
                        and ray_error >= 0.48
                    )
                )
            )

        score = (
            min(
                projection['pixel_distance']
                / max(
                    projection['strong_radius'],
                    1.0,
                ),
                4.0,
            )
            if projection['projectable']
            else 2.0
        )
        score += min(
            fit_distance
            / max(
                self.direct_merge_distance,
                1.0e-6,
            ),
            4.0,
        )
        score += 0.60 * min(
            ray_error
            / max(
                self.ray_match_threshold,
                1.0e-6,
            ),
            4.0,
        )
        if math.isfinite(appearance_distance):
            score += 0.90 * min(
                appearance_distance
                / max(
                    self.appearance_match_threshold,
                    1.0e-6,
                ),
                4.0,
            )
        if bundle_strong:
            score -= 1.0
        if appearance_guided:
            score -= 0.75

        return {
            'strong': bool(strong),
            'ambiguous': bool(ambiguous),
            'distinct': bool(distinct),
            'cannot_link': False,
            'score': float(score),
            'reason': (
                'projection'
                if projection_strong
                else (
                    'position'
                    if geometry_strong
                    else (
                        'bearing'
                        if bearing_strong
                        else (
                            'bundle'
                            if bundle_strong
                            else 'appearance'
                        )
                    )
                )
            ),
            'fit_distance': fit_distance,
            'ground_distance': ground_distance,
            'ray_error': ray_error,
            'appearance_distance': appearance_distance,
            'bundle': bundle,
            'projection': projection,
            'geometry_update_ok': (
                fit_distance <= 0.30
                or (
                    bundle_strong
                    and ray_error <= 0.18
                )
            ),
        }

    def _candidate_match_cost(
        self,
        tracklet,
        candidate,
        current_keys,
        cannot_link_pairs,
    ):
        active_candidate = self._active_tracklets(
            candidate.track_keys,
            current_keys,
        )
        if self._has_cannot_link(
            {tracklet.key},
            {item.key for item in active_candidate},
            cannot_link_pairs,
        ):
            return self.INVALID_COST

        track_fit = tracklet.estimate(
            self.ray_inlier_threshold,
            self.minimum_baseline,
        )
        candidate_fit = candidate.estimate(
            self.ray_inlier_threshold,
            self.minimum_baseline,
        )
        fit_distance = (
            self._distance(
                (
                    track_fit['x'],
                    track_fit['y'],
                ),
                (
                    candidate_fit['x'],
                    candidate_fit['y'],
                ),
            )
            if (
                track_fit is not None
                and candidate_fit is not None
            )
            else float('inf')
        )
        track_ground = tracklet.ground_center()
        candidate_ground = candidate.ground_center()
        ground_distance = (
            self._distance(
                track_ground,
                candidate_ground,
            )
            if (
                track_ground is not None
                and candidate_ground is not None
            )
            else float('inf')
        )
        appearance_distance = (
            tracklet.appearance_distance_to(
                candidate.appearance_bank)
        )
        appearance_ok = (
            appearance_distance
            <= self.appearance_ambiguity_threshold
        )
        bundle_ok, _bundle = (
            self._combined_samples_compatible(
                tracklet.samples,
                candidate.samples,
            )
        )
        duplicate_like = False
        for active in active_candidate:
            iou = self._bbox_iou(
                tracklet.last_bbox,
                active.last_bbox,
            )
            pixel_distance = self._distance(
                tracklet.last_image_center,
                active.last_image_center,
            )
            if (
                iou >= self.duplicate_minimum_bbox_iou
                or pixel_distance
                <= self.duplicate_maximum_pixel_distance
            ):
                duplicate_like = True
                break

        # StrongSORT AFLink uses temporal/spatial gates before bipartite
        # assignment.  Here appearance is an additional gate, never the sole
        # reason to fuse two spatially separate candidates.
        compatible = (
            fit_distance <= self.candidate_spatial_gate
            or ground_distance <= 0.46
            or bundle_ok
            or duplicate_like
            or (
                appearance_ok
                and (
                    fit_distance <= 0.66
                    or ground_distance <= 0.66
                )
            )
        )
        if not compatible:
            return self.INVALID_COST

        appearance_cost = (
            min(
                appearance_distance
                / max(
                    self.appearance_match_threshold,
                    1.0e-6,
                ),
                3.0,
            )
            if math.isfinite(appearance_distance)
            else 2.0
        )
        return float(
            min(fit_distance, 1.0)
            + 0.50 * min(ground_distance, 1.0)
            + 0.65 * appearance_cost
            + (0.0 if bundle_ok else 0.25)
            + (0.0 if duplicate_like else 0.15)
        )

    def _new_candidate(self, tracklet):
        candidate_id = self._next_candidate_id
        self._next_candidate_id += 1
        candidate = _DeferredLandmarkCandidate(
            candidate_id=candidate_id,
            tracklet=tracklet,
            sample_window=self.sample_window * 3,
        )
        self._candidates[candidate_id] = candidate
        self._track_to_candidate[
            tracklet.key
        ] = candidate_id
        self.logger.info(
            f'YOLO deferred candidate: '
            f'candidate={candidate_id}, '
            f'drone={tracklet.source_id}, '
            f'local_track='
            f'{tracklet.local_track_id}')
        return candidate

    def _remove_candidate(self, candidate_id):
        candidate = self._candidates.pop(
            candidate_id,
            None,
        )
        if candidate is None:
            return
        for key in list(candidate.track_keys):
            if (
                self._track_to_candidate.get(key)
                == candidate_id
            ):
                self._track_to_candidate.pop(
                    key, None)

    def _assign_unmapped_to_candidates(
        self,
        tracklets,
        current_keys,
        cannot_link_pairs,
    ):
        if not tracklets:
            return

        grouped = {}
        for tracklet in tracklets:
            grouped.setdefault(
                (
                    tracklet.source_id,
                    tracklet.class_id,
                ),
                [],
            ).append(tracklet)

        for group_key, group_tracklets in grouped.items():
            candidates = self._candidates_for(
                group_key[0],
                group_key[1],
            )
            rows = [
                tracklet
                for tracklet in group_tracklets
                if tracklet.key
                not in self._track_to_candidate
            ]
            if rows and candidates:
                cost_matrix = np.full(
                    (
                        len(rows),
                        len(candidates),
                    ),
                    self.INVALID_COST,
                    dtype=np.float64,
                )
                for row_index, tracklet in enumerate(rows):
                    for column_index, candidate in enumerate(
                        candidates
                    ):
                        cost_matrix[
                            row_index,
                            column_index,
                        ] = self._candidate_match_cost(
                            tracklet,
                            candidate,
                            current_keys,
                            cannot_link_pairs,
                        )

                assignments = self._hungarian(
                    cost_matrix,
                    unmatched_cost=50.0,
                )
                assigned_rows = set()
                for row_index, column_index in assignments:
                    tracklet = rows[row_index]
                    candidate = candidates[column_index]
                    candidate.absorb(tracklet)
                    self._track_to_candidate[
                        tracklet.key
                    ] = candidate.candidate_id
                    assigned_rows.add(row_index)

                for row_index, tracklet in enumerate(rows):
                    if row_index not in assigned_rows:
                        self._new_candidate(
                            tracklet)
            else:
                for tracklet in rows:
                    self._new_candidate(
                        tracklet)

        # Existing candidate members still need their newest sample appended.
        for tracklet in tracklets:
            candidate_id = self._track_to_candidate.get(
                tracklet.key)
            if candidate_id is None:
                continue
            candidate = self._candidates.get(
                candidate_id)
            if candidate is not None:
                candidate.absorb(tracklet)

    def _candidate_mature(
        self,
        candidate,
        now,
    ):
        if (
            len(candidate.samples)
            < max(
                self.confirmation_hits + 1,
                self.minimum_triangulation_inliers + 1,
                5,
            )
        ):
            return False, None
        if (
            float(now) - candidate.first_seen
            < self.minimum_confirmation_age
        ):
            return False, None

        fit = candidate.estimate(
            self.ray_inlier_threshold,
            self.minimum_baseline,
        )
        if fit is None:
            return False, None
        mature = (
            fit['inliers']
            >= self.minimum_triangulation_inliers
            and fit['residual']
            <= self.maximum_ray_residual
            and fit['baseline']
            >= self.minimum_baseline
        )
        return bool(mature), fit

    def _absorb_candidate_geometry(
        self,
        candidate,
        landmark,
        fit,
        update_geometry,
    ):
        for key in candidate.track_keys:
            self._track_to_marker[
                key
            ] = landmark.marker_id
            landmark.track_keys.add(key)

        landmark.last_seen = max(
            landmark.last_seen,
            candidate.last_seen,
        )
        for feature in candidate.appearance_bank:
            _append_normalized_feature(
                landmark.appearance_bank,
                feature,
            )
        if update_geometry:
            for sample in candidate.samples:
                landmark.samples.append(sample)
            candidate_point = np.asarray(
                [fit['x'], fit['y']],
                dtype=np.float64,
            )
            anchor = landmark.anchor_center()
            if (
                anchor is None
                or float(np.linalg.norm(
                    candidate_point - anchor
                )) <= 0.32
            ):
                landmark.anchor_points.append(
                    candidate_point)

    def _merge_candidate_into_landmark(
        self,
        candidate,
        landmark,
        relation,
    ):
        fit = candidate.estimate(
            self.ray_inlier_threshold,
            self.minimum_baseline,
        )
        if fit is None:
            return
        self._absorb_candidate_geometry(
            candidate,
            landmark,
            fit,
            relation['geometry_update_ok'],
        )
        self.logger.info(
            f'YOLO projection/AFLink revisit: '
            f'candidate={candidate.candidate_id}, '
            f'drone={candidate.source_id} -> '
            f'marker {landmark.marker_id}, '
            f'reason={relation["reason"]}, '
            f'fit_distance='
            f'{relation.get("fit_distance", float("inf")):.2f}m, '
            f'ray_error='
            f'{relation.get("ray_error", float("inf")):.2f}m, '
            f'appearance='
            f'{relation.get("appearance_distance", float("inf")):.3f}, '
            f'geometry_update='
            f'{"accepted" if relation["geometry_update_ok"] else "skipped"}')
        self._remove_candidate(
            candidate.candidate_id)

    def _attach_tracklet(
        self,
        tracklet,
        landmark,
        relation,
    ):
        candidate_id = self._track_to_candidate.get(
            tracklet.key)
        if candidate_id is not None:
            candidate = self._candidates.get(
                candidate_id)
            if candidate is not None:
                self._merge_candidate_into_landmark(
                    candidate,
                    landmark,
                    relation,
                )
                return

        self._track_to_marker[
            tracklet.key
        ] = landmark.marker_id
        fit = tracklet.estimate(
            self.ray_inlier_threshold,
            self.minimum_baseline,
        )
        landmark.absorb_tracklet(
            tracklet,
            fit=fit,
            update_geometry=(
                relation['geometry_update_ok']),
        )
        self.logger.info(
            f'YOLO projection revisit linked: '
            f'{tracklet.source_id}:'
            f'{tracklet.local_track_id} -> '
            f'marker {landmark.marker_id}, '
            f'reason={relation["reason"]}, '
            f'fit_distance='
            f'{relation.get("fit_distance", float("inf")):.2f}m, '
            f'ray_error='
            f'{relation.get("ray_error", float("inf")):.2f}m, '
            f'appearance='
            f'{relation.get("appearance_distance", float("inf")):.3f}, '
            f'geometry_update='
            f'{"accepted" if relation["geometry_update_ok"] else "skipped"}')

    def _create_landmark_from_candidate(
        self,
        candidate,
        fit,
    ):
        representative = None
        for key in candidate.track_keys:
            representative = self._tracklets.get(
                key)
            if representative is not None:
                break
        if representative is None:
            return None

        marker_id = self._next_marker_id
        self._next_marker_id += 1
        landmark = _FastLandmark(
            marker_id=marker_id,
            tracklet=representative,
            sample_window=self.sample_window * 3,
            initial_fit=fit,
        )
        landmark.track_keys = set(
            candidate.track_keys)
        landmark.samples = deque(
            candidate.samples,
            maxlen=max(
                24,
                self.sample_window * 3,
            ),
        )
        landmark.anchor_points = deque(
            [
                np.asarray(
                    [fit['x'], fit['y']],
                    dtype=np.float64,
                )
            ],
            maxlen=7,
        )
        landmark.appearance_bank = deque(
            list(candidate.appearance_bank),
            maxlen=48,
        )
        landmark.last_seen = candidate.last_seen
        self._landmarks[
            marker_id
        ] = landmark
        for key in candidate.track_keys:
            self._track_to_marker[
                key
            ] = marker_id

        center = landmark.center(
            self.ray_inlier_threshold,
            self.minimum_baseline,
        )
        self.logger.info(
            f'YOLO object confirmed: '
            f'marker={marker_id}, '
            f'drone={candidate.source_id}, '
            f'candidate={candidate.candidate_id}, '
            f'tracklets={len(candidate.track_keys)}, '
            f'hits={len(candidate.samples)}, '
            f'inliers={fit["inliers"]}, '
            f'residual={fit["residual"]:.2f}m, '
            f'position=('
            f'{center[0]:.2f}, '
            f'{center[1]:.2f}, 0.00)')
        self._remove_candidate(
            candidate.candidate_id)
        return {
            'marker_id': marker_id,
            'x': float(center[0]),
            'y': float(center[1]),
            'z': 0.0,
        }

    def _expire(self, now):
        expired_tracklets = [
            key
            for key, tracklet
            in self._tracklets.items()
            if (
                float(now) - tracklet.last_seen
                > self.tracklet_timeout
            )
        ]
        for key in expired_tracklets:
            self._tracklets.pop(
                key, None)
            # Keep marker/candidate membership until the component expires;
            # AFLink-style fragmented IDs may reappear through a new key.

        expired_candidates = [
            candidate_id
            for candidate_id, candidate
            in self._candidates.items()
            if (
                float(now) - candidate.last_seen
                > self.candidate_timeout
            )
        ]
        for candidate_id in expired_candidates:
            self._remove_candidate(
                candidate_id)

    def _strong_assignment(
        self,
        tracklets,
        landmarks,
        projection_context,
        current_keys,
        cannot_link_pairs,
    ):
        """DeepSORT-style matching cascade with one-to-one Hungarian stages.

        Recently observed landmarks are matched first.  Older landmarks remain
        available in later cascade levels for long lawnmower revisits, while a
        stale identity cannot steal a detection from a fresh landmark.
        """
        if not tracklets or not landmarks:
            return []

        remaining_rows = list(range(len(tracklets)))
        used_columns = set()
        output = []
        newest_time = max(
            tracklet.last_seen
            for tracklet in tracklets
        )
        cascade_limits = (
            2.0,
            8.0,
            30.0,
            float('inf'),
        )
        previous_limit = -1.0

        for age_limit in cascade_limits:
            if not remaining_rows:
                break

            eligible_columns = [
                index
                for index, landmark in enumerate(landmarks)
                if (
                    index not in used_columns
                    and previous_limit
                    < newest_time - landmark.last_seen
                    <= age_limit
                )
            ]
            previous_limit = age_limit
            if not eligible_columns:
                continue

            cost_matrix = np.full(
                (
                    len(remaining_rows),
                    len(eligible_columns),
                ),
                self.INVALID_COST,
                dtype=np.float64,
            )
            relations = {}
            for local_row, row_index in enumerate(
                remaining_rows
            ):
                tracklet = tracklets[row_index]
                for local_column, column_index in enumerate(
                    eligible_columns
                ):
                    landmark = landmarks[column_index]
                    relation = self._tracklet_landmark_relation(
                        tracklet,
                        landmark,
                        projection_context,
                        current_keys,
                        cannot_link_pairs,
                    )
                    relations[
                        (local_row, local_column)
                    ] = relation
                    if relation['strong']:
                        cost_matrix[
                            local_row,
                            local_column,
                        ] = relation['score']

            assignments = self._hungarian(
                cost_matrix,
                unmatched_cost=50.0,
            )
            matched_rows = set()
            for local_row, local_column in assignments:
                row_index = remaining_rows[local_row]
                column_index = eligible_columns[
                    local_column]
                relation = relations[
                    (local_row, local_column)]
                output.append(
                    (
                        tracklets[row_index],
                        landmarks[column_index],
                        relation,
                    )
                )
                matched_rows.add(row_index)
                used_columns.add(column_index)

            remaining_rows = [
                row_index
                for row_index in remaining_rows
                if row_index not in matched_rows
            ]

        return output

    def update_frame(
        self,
        source_id,
        detections,
        now,
        projection_context=None,
    ):
        with self._lock:
            self._expire(now)

            current_tracklets = []
            current_keys = set()
            for detection in detections:
                key = (
                    str(source_id),
                    int(detection['track_id']),
                )
                tracklet = self._tracklets.get(
                    key)
                if tracklet is None:
                    tracklet = _FastTracklet(
                        source_id=source_id,
                        local_track_id=(
                            detection['track_id']),
                        class_id=(
                            detection['class_id']),
                        class_name=(
                            detection['class_name']),
                        sample_window=(
                            self.sample_window),
                    )
                    self._tracklets[key] = (
                        tracklet)

                tracklet.update(
                    detection,
                    now,
                )
                current_tracklets.append(
                    tracklet)
                current_keys.add(
                    tracklet.key)

            cannot_link_pairs = (
                self._distinct_pairs(
                    current_tracklets)
            )
            confirmed = []

            # Existing mapped tracks only update geometry under a strict gate.
            for tracklet in current_tracklets:
                marker_id = self._track_to_marker.get(
                    tracklet.key)
                if marker_id is None:
                    continue
                landmark = self._landmarks.get(
                    marker_id)
                if landmark is None:
                    continue
                anchor = landmark.center(
                    self.ray_inlier_threshold,
                    self.minimum_baseline,
                )
                fit = tracklet.estimate(
                    self.ray_inlier_threshold,
                    self.minimum_baseline,
                )
                fit_distance = (
                    self._distance(
                        (fit['x'], fit['y']),
                        anchor,
                    )
                    if fit is not None
                    else float('inf')
                )
                ray_error = tracklet.ray_error(
                    anchor,
                    recent=10,
                )
                landmark.absorb_tracklet(
                    tracklet,
                    fit=fit,
                    update_geometry=(
                        fit_distance <= 0.30
                        or ray_error <= 0.16
                    ),
                )

            unmapped = [
                tracklet
                for tracklet in current_tracklets
                if tracklet.key
                not in self._track_to_marker
            ]

            # ORB-SLAM/UCMC-style projection + one-to-one assignment.
            grouped = {}
            for tracklet in unmapped:
                grouped.setdefault(
                    (
                        tracklet.source_id,
                        tracklet.class_id,
                    ),
                    [],
                ).append(tracklet)

            for group_key, group_tracklets in grouped.items():
                landmarks = self._landmarks_for(
                    group_key[0],
                    group_key[1],
                )
                assignments = self._strong_assignment(
                    group_tracklets,
                    landmarks,
                    projection_context,
                    current_keys,
                    cannot_link_pairs,
                )
                for tracklet, landmark, relation in assignments:
                    if (
                        tracklet.key
                        in self._track_to_marker
                    ):
                        continue
                    self._attach_tracklet(
                        tracklet,
                        landmark,
                        relation,
                    )

            remaining = [
                tracklet
                for tracklet in unmapped
                if tracklet.key
                not in self._track_to_marker
            ]
            self._assign_unmapped_to_candidates(
                remaining,
                current_keys,
                cannot_link_pairs,
            )

            active_candidate_ids = {
                self._track_to_candidate[key]
                for key in current_keys
                if key in self._track_to_candidate
            }
            for candidate_id in list(
                active_candidate_ids
            ):
                candidate = self._candidates.get(
                    candidate_id)
                if candidate is None:
                    continue
                active_tracklets = self._active_tracklets(
                    candidate.track_keys,
                    current_keys,
                )
                if not active_tracklets:
                    continue
                active_tracklet = max(
                    active_tracklets,
                    key=lambda item: item.last_seen,
                )
                landmarks = self._landmarks_for(
                    candidate.source_id,
                    candidate.class_id,
                )

                best_landmark = None
                best_relation = None
                relations = []
                for landmark in landmarks:
                    relation = (
                        self._candidate_landmark_relation(
                            candidate,
                            active_tracklet,
                            landmark,
                            projection_context,
                            current_keys,
                            cannot_link_pairs,
                        )
                    )
                    relations.append(
                        (landmark, relation))
                    if relation['strong']:
                        if (
                            best_relation is None
                            or relation['score']
                            < best_relation['score']
                        ):
                            best_landmark = landmark
                            best_relation = relation

                if best_landmark is not None:
                    self._merge_candidate_into_landmark(
                        candidate,
                        best_landmark,
                        best_relation,
                    )
                    continue

                mature, fit = self._candidate_mature(
                    candidate,
                    now,
                )
                if not mature:
                    continue

                if not landmarks:
                    result = (
                        self._create_landmark_from_candidate(
                            candidate,
                            fit,
                        )
                    )
                    if result is not None:
                        confirmed.append(result)
                    continue

                all_distinct = True
                minimum_counter = None
                for landmark, relation in relations:
                    marker_id = landmark.marker_id
                    previous = candidate.distinct_evidence.get(
                        marker_id, 0)
                    if relation['cannot_link']:
                        current = max(
                            previous + 1,
                            self.distinct_evidence_frames,
                        )
                    elif relation['distinct']:
                        current = previous + 1
                    else:
                        current = 0

                    candidate.distinct_evidence[
                        marker_id
                    ] = current
                    minimum_counter = (
                        current
                        if minimum_counter is None
                        else min(
                            minimum_counter,
                            current,
                        )
                    )
                    if (
                        current
                        < self.distinct_evidence_frames
                    ):
                        all_distinct = False

                if not all_distinct:
                    if not candidate.hold_logged:
                        candidate.hold_logged = True
                        self.logger.info(
                            f'YOLO duplicate-veto hold: '
                            f'candidate='
                            f'{candidate.candidate_id}, '
                            f'drone={candidate.source_id}; '
                            f'existing landmark remains '
                            f'projection/bearing ambiguous')
                    continue

                # Final defensive veto.  A new marker is never created inside
                # the direct sub-0.4 m gate unless simultaneous cannot-link
                # evidence already proved two separate boxes.
                fit_center = np.asarray(
                    [fit['x'], fit['y']],
                    dtype=np.float64,
                )
                too_close = False
                has_cannot_link = False
                for landmark, relation in relations:
                    if relation['cannot_link']:
                        has_cannot_link = True
                    landmark_center = landmark.center(
                        self.ray_inlier_threshold,
                        self.minimum_baseline,
                    )
                    if (
                        self._distance(
                            fit_center,
                            landmark_center,
                        )
                        < self.new_object_minimum_separation
                    ):
                        too_close = True

                if too_close and not has_cannot_link:
                    continue

                result = (
                    self._create_landmark_from_candidate(
                        candidate,
                        fit,
                    )
                )
                if result is not None:
                    confirmed.append(result)

            return confirmed



class UncertaintyAwareLandmarkRegistry(RealtimeLandmarkRegistry):
    """Open-world landmark association with statistical gates.

    This class keeps the v12/v14 real-time pipeline and replaces the hand-tuned
    identity rules with four fixed, interpretable mechanisms:

    1. covariance-aware Mahalanobis association (UCMC/DeepSORT pattern),
    2. ORB-SLAM-style reprojection consistency,
    3. DeepSORT nearest-neighbour appearance gallery,
    4. AFLink deferred tracklet components.

    Bearing or ray-bundle agreement is supporting evidence only.  Neither can
    merge identities by itself.  A high-information three-ray candidate can be
    committed immediately when every existing landmark is statistically
    incompatible, so a briefly visible new object does not wait for an
    arbitrary hit counter.
    """

    CHI2_SAME_2D = 5.991       # 95 %
    CHI2_COMPATIBLE_2D = 9.210  # 99 %
    CHI2_DISTINCT_2D = 13.816   # 99.9 %
    CHI2_VERY_DISTINCT_2D = 25.0

    def __init__(self, *args, marker_size=0.14, **kwargs):
        super().__init__(*args, **kwargs)
        self.marker_size = max(0.05, float(marker_size))

        # These are derived physical/statistical policies, not per-layout
        # tuning knobs.  YAML values remain accepted for launch compatibility.
        self.minimum_baseline = max(0.025, 0.18 * self.marker_size)
        self.minimum_confirmation_age = 0.10
        self.minimum_triangulation_inliers = 3
        self.confirmation_hits = 2
        self.distinct_evidence_frames = 1
        self.new_object_minimum_separation = 0.0

        # Descriptor thresholds belong to this exact normalized descriptor and
        # therefore do not need to be re-tuned for each arena layout.
        self.appearance_match_threshold = 0.18
        self.appearance_ambiguity_threshold = 0.32

        # Candidate precision needed for immediate publication.  It scales with
        # the physical target size while retaining tolerance for AI-deck pose
        # and calibration noise.
        self.fast_commit_radius95 = max(
            0.16,
            1.35 * self.marker_size,
        )
        self.maximum_commit_radius95 = max(
            0.24,
            1.90 * self.marker_size,
        )

    def _landmarks_for(self, source_id, class_id):
        # World identities are global.  Filtering by drone creates one marker
        # per camera for the same physical object.
        del source_id
        return [
            landmark
            for landmark in self._landmarks.values()
            if landmark.class_id == int(class_id)
        ]

    def _candidates_for(self, source_id, class_id):
        # AFLink components are also global so cf6/cf7 fragments can converge
        # before a marker is published.
        del source_id
        return [
            candidate
            for candidate in self._candidates.values()
            if candidate.class_id == int(class_id)
        ]

    @staticmethod
    def _regularize_covariance(covariance, floor_sigma=0.035):
        covariance = np.asarray(covariance, dtype=np.float64)
        if covariance.shape != (2, 2):
            covariance = np.eye(2, dtype=np.float64) * floor_sigma ** 2
        covariance = np.nan_to_num(
            covariance,
            nan=floor_sigma ** 2,
            posinf=0.36,
            neginf=0.36,
        )
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.clip(
            eigenvalues,
            floor_sigma ** 2,
            0.60 ** 2,
        )
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def _fallback_distribution(self, entity):
        points = np.asarray(
            [(sample[0], sample[1]) for sample in entity.samples],
            dtype=np.float64,
        )
        if len(points) == 0:
            return None
        center = np.median(points, axis=0)
        if len(points) >= 2:
            delta = points - center[None, :]
            covariance = np.cov(delta.T, bias=False)
            if np.asarray(covariance).shape != (2, 2):
                covariance = np.eye(2) * 0.08 ** 2
            covariance = covariance / max(1.0, float(len(points)))
        else:
            covariance = np.eye(2) * 0.20 ** 2

        ranges = []
        for sample in entity.samples:
            ranges.append(math.hypot(
                float(sample[0]) - float(sample[4]),
                float(sample[1]) - float(sample[5]),
            ))
        median_range = float(np.median(ranges)) if ranges else 0.0
        floor = 0.035 + 0.035 * min(4.0, median_range)
        covariance = self._regularize_covariance(
            covariance + np.eye(2) * floor * floor,
            floor_sigma=floor,
        )
        radius95 = float(math.sqrt(
            self.CHI2_SAME_2D
            * float(np.max(np.linalg.eigvalsh(covariance)))
        ))
        return {
            'mean': center,
            'covariance': covariance,
            'radius95': radius95,
            'fit': None,
        }

    def _entity_distribution(self, entity, landmark=False):
        fit = entity.estimate(
            self.ray_inlier_threshold,
            self.minimum_baseline,
        )
        fallback = self._fallback_distribution(entity)
        if fit is None:
            if landmark and fallback is not None and hasattr(entity, 'anchor_center'):
                anchor = entity.anchor_center()
                if anchor is not None:
                    fallback['mean'] = np.asarray(anchor, dtype=np.float64)
            return fallback

        mean = np.asarray([fit['x'], fit['y']], dtype=np.float64)
        covariance = self._regularize_covariance(
            fit.get('covariance', np.eye(2) * 0.10 ** 2)
        )

        if fallback is not None:
            covariance = self._regularize_covariance(
                covariance + 0.35 * fallback['covariance']
            )
        if landmark and hasattr(entity, 'anchor_center'):
            anchor = entity.anchor_center()
            if anchor is not None:
                mean = np.asarray(anchor, dtype=np.float64)
                if len(entity.anchor_points) >= 2:
                    anchors = np.asarray(entity.anchor_points, dtype=np.float64)
                    anchor_delta = anchors - np.median(anchors, axis=0)[None, :]
                    anchor_cov = np.cov(anchor_delta.T, bias=False)
                    if np.asarray(anchor_cov).shape == (2, 2):
                        covariance = self._regularize_covariance(
                            covariance + anchor_cov / max(1.0, len(anchors))
                        )

        radius95 = float(math.sqrt(
            self.CHI2_SAME_2D
            * float(np.max(np.linalg.eigvalsh(covariance)))
        ))
        return {
            'mean': mean,
            'covariance': covariance,
            'radius95': radius95,
            'fit': fit,
        }

    @staticmethod
    def _mahalanobis_squared(first, second):
        if first is None or second is None:
            return float('inf'), float('inf')
        delta = np.asarray(first['mean']) - np.asarray(second['mean'])
        covariance = (
            np.asarray(first['covariance'])
            + np.asarray(second['covariance'])
        )
        covariance = 0.5 * (covariance + covariance.T)
        try:
            inverse = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(covariance)
        distance2 = float(delta.T @ inverse @ delta)
        metric_distance = float(np.linalg.norm(delta))
        return distance2, metric_distance

    @staticmethod
    def _projection_nis(projection, tracklet):
        if not projection.get('projectable', False):
            return float('inf')
        if tracklet.last_bbox is None:
            return float('inf')
        _x, _y, width, height = tracklet.last_bbox
        # Detector centre noise grows with box size and edge truncation.  This
        # produces a normalized residual instead of a fixed 20/40-pixel gate.
        sigma_px = max(7.0, 0.28 * max(width, height))
        return float(
            (projection['pixel_distance'] / sigma_px) ** 2
        )

    def _relation_core(
        self,
        entity,
        active_tracklet,
        landmark,
        projection_context,
        cannot_link=False,
    ):
        if cannot_link:
            return {
                'strong': False,
                'ambiguous': False,
                'distinct': True,
                'cannot_link': True,
                'score': self.INVALID_COST,
                'reason': 'cannot-link',
                'geometry_update_ok': False,
                'appearance_distance': float('inf'),
                'mahalanobis2': float('inf'),
            }

        observation = self._entity_distribution(entity, landmark=False)
        landmark_distribution = self._entity_distribution(
            landmark, landmark=True
        )
        mahalanobis2, metric_distance = self._mahalanobis_squared(
            observation,
            landmark_distribution,
        )
        landmark_center = landmark_distribution['mean']
        ray_error = entity.ray_error(landmark_center, recent=14)
        appearance_distance = entity.appearance_distance_to(
            landmark.appearance_bank
        )
        projection = self._projection_relation(
            active_tracklet,
            landmark,
            projection_context,
        )
        projection_nis = self._projection_nis(
            projection,
            active_tracklet,
        )
        bundle_ok, bundle = self._combined_samples_compatible(
            entity.samples,
            landmark.samples,
        )

        metric_same = mahalanobis2 <= self.CHI2_SAME_2D
        metric_compatible = mahalanobis2 <= self.CHI2_COMPATIBLE_2D
        metric_distinct = mahalanobis2 >= self.CHI2_DISTINCT_2D
        metric_very_distinct = (
            mahalanobis2 >= self.CHI2_VERY_DISTINCT_2D
        )
        projection_same = projection_nis <= self.CHI2_SAME_2D
        projection_compatible = (
            projection_nis <= self.CHI2_COMPATIBLE_2D
        )
        projection_distinct = (
            projection.get('available', False)
            and (
                projection.get('far', False)
                or projection_nis >= self.CHI2_DISTINCT_2D
            )
        )
        appearance_same = (
            math.isfinite(appearance_distance)
            and appearance_distance <= self.appearance_match_threshold
        )
        appearance_compatible = (
            math.isfinite(appearance_distance)
            and appearance_distance <= self.appearance_ambiguity_threshold
        )
        bundle_support = (
            bundle_ok
            and ray_error <= max(0.16, 1.2 * self.marker_size)
        )
        bearing_support = (
            ray_error <= max(0.12, 0.9 * self.marker_size)
        )

        # A merge always needs two independent families of evidence.  Bearing
        # and bundle fit are never sufficient on their own.
        strong = (
            (
                metric_same
                and (
                    projection_compatible
                    or appearance_compatible
                    or bundle_support
                )
            )
            or (
                projection_same
                and (
                    appearance_compatible
                    or metric_compatible
                )
            )
            or (
                appearance_same
                and metric_compatible
                and not projection_distinct
            )
        )

        combined_radius = (
            (observation['radius95'] if observation else 0.30)
            + (landmark_distribution['radius95'] if landmark_distribution else 0.30)
        )
        physical_margin = metric_distance - combined_radius
        physically_separated = (
            physical_margin > max(self.marker_size, 0.10)
        )

        # A new object is declared only when position uncertainty and current
        # reprojection both reject the old landmark.  Strong appearance can
        # delay a marginal decision, but cannot override a very decisive metric
        # plus reprojection separation (identical-looking targets are allowed).
        distinct = (
            not strong
            and metric_distinct
            and physically_separated
            and (
                projection_distinct
                or not projection.get('available', False)
            )
            and (
                not appearance_same
                or metric_very_distinct
            )
        )
        ambiguous = not strong and not distinct

        score = min(mahalanobis2, 50.0)
        if math.isfinite(projection_nis):
            score += 0.60 * min(projection_nis, 50.0)
        if math.isfinite(appearance_distance):
            score += 4.0 * appearance_distance
        if bundle_support:
            score -= 0.5
        if bearing_support:
            score -= 0.2

        geometry_update_ok = (
            metric_same
            and (
                projection_compatible
                or appearance_compatible
            )
            and observation is not None
            and observation['radius95'] <= self.maximum_commit_radius95
        )
        if projection_same and appearance_same:
            reason = 'projection+appearance'
        elif metric_same and projection_compatible:
            reason = 'mahalanobis+projection'
        elif metric_same and appearance_compatible:
            reason = 'mahalanobis+appearance'
        elif metric_same and bundle_support:
            reason = 'mahalanobis+bundle'
        else:
            reason = 'uncertainty'

        fit_distance = metric_distance
        ground_center = entity.ground_center()
        ground_distance = (
            self._distance(ground_center, landmark_center)
            if ground_center is not None
            else float('inf')
        )
        return {
            'strong': bool(strong),
            'ambiguous': bool(ambiguous),
            'distinct': bool(distinct),
            'cannot_link': False,
            'score': float(score),
            'reason': reason,
            'fit_distance': float(fit_distance),
            'ground_distance': float(ground_distance),
            'ray_error': float(ray_error),
            'appearance_distance': float(appearance_distance),
            'bundle': bundle,
            'projection': projection,
            'projection_nis': float(projection_nis),
            'mahalanobis2': float(mahalanobis2),
            'observation_radius95': (
                float(observation['radius95'])
                if observation is not None
                else float('inf')
            ),
            'geometry_update_ok': bool(geometry_update_ok),
        }

    def _tracklet_landmark_relation(
        self,
        tracklet,
        landmark,
        projection_context,
        current_keys,
        cannot_link_pairs,
    ):
        active_landmark = self._active_tracklets(
            landmark.track_keys,
            current_keys,
        )
        cannot_link = self._has_cannot_link(
            {tracklet.key},
            {item.key for item in active_landmark},
            cannot_link_pairs,
        )
        return self._relation_core(
            tracklet,
            tracklet,
            landmark,
            projection_context,
            cannot_link=cannot_link,
        )

    def _candidate_landmark_relation(
        self,
        candidate,
        active_tracklet,
        landmark,
        projection_context,
        current_keys,
        cannot_link_pairs,
    ):
        active_landmark = self._active_tracklets(
            landmark.track_keys,
            current_keys,
        )
        active_candidate = self._active_tracklets(
            candidate.track_keys,
            current_keys,
        )
        cannot_link = self._has_cannot_link(
            {item.key for item in active_candidate},
            {item.key for item in active_landmark},
            cannot_link_pairs,
        )
        return self._relation_core(
            candidate,
            active_tracklet,
            landmark,
            projection_context,
            cannot_link=cannot_link,
        )

    def _candidate_match_cost(
        self,
        tracklet,
        candidate,
        current_keys,
        cannot_link_pairs,
    ):
        active_candidate = self._active_tracklets(
            candidate.track_keys,
            current_keys,
        )
        if self._has_cannot_link(
            {tracklet.key},
            {item.key for item in active_candidate},
            cannot_link_pairs,
        ):
            return self.INVALID_COST

        track_distribution = self._entity_distribution(tracklet)
        candidate_distribution = self._entity_distribution(candidate)
        mahalanobis2, metric_distance = self._mahalanobis_squared(
            track_distribution,
            candidate_distribution,
        )
        appearance_distance = tracklet.appearance_distance_to(
            candidate.appearance_bank
        )
        appearance_same = (
            math.isfinite(appearance_distance)
            and appearance_distance <= self.appearance_match_threshold
        )
        appearance_compatible = (
            math.isfinite(appearance_distance)
            and appearance_distance <= self.appearance_ambiguity_threshold
        )
        bundle_ok, _bundle = self._combined_samples_compatible(
            tracklet.samples,
            candidate.samples,
        )

        duplicate_like = False
        for active in active_candidate:
            iou = self._bbox_iou(tracklet.last_bbox, active.last_bbox)
            pixel_distance = self._distance(
                tracklet.last_image_center,
                active.last_image_center,
            )
            if (
                iou >= self.duplicate_minimum_bbox_iou
                or pixel_distance <= self.duplicate_maximum_pixel_distance
            ):
                duplicate_like = True
                break

        compatible = (
            mahalanobis2 <= self.CHI2_COMPATIBLE_2D
            and (
                appearance_compatible
                or duplicate_like
                or bundle_ok
            )
        ) or (
            appearance_same
            and mahalanobis2 <= self.CHI2_DISTINCT_2D
        )
        if not compatible:
            return self.INVALID_COST

        appearance_cost = (
            4.0 * appearance_distance
            if math.isfinite(appearance_distance)
            else 1.2
        )
        return float(
            min(mahalanobis2, 30.0)
            + appearance_cost
            + 0.5 * min(metric_distance, 1.0)
            + (0.0 if duplicate_like else 0.2)
            + (0.0 if bundle_ok else 0.2)
        )

    def _candidate_mature(self, candidate, now):
        # Three valid, geometrically separated rays are the mathematical
        # minimum for a robust 2-D line intersection.  No arbitrary five-hit
        # floor is used.
        if len(candidate.samples) < 3:
            return False, None
        if float(now) - candidate.first_seen < self.minimum_confirmation_age:
            return False, None
        fit = candidate.estimate(
            self.ray_inlier_threshold,
            self.minimum_baseline,
        )
        if fit is None:
            return False, None

        inlier_ratio = fit['inliers'] / max(1.0, float(len(candidate.samples)))
        residual_limit = max(
            0.12,
            1.35 * self.marker_size,
        )
        mature = (
            fit['inliers'] >= 3
            and inlier_ratio >= 0.60
            and fit['residual'] <= residual_limit
            and fit['baseline'] >= self.minimum_baseline
            and fit.get('radius95', float('inf')) <= self.maximum_commit_radius95
        )
        return bool(mature), fit

    PUBLISH_SUPPRESSION_RADIUS_M = 0.30

    def _create_landmark_from_candidate(self, candidate, fit):
        """Final deterministic publication firewall.

        No new marker ID is ever published within 30 cm of an already
        published landmark of the same class.  This is intentionally checked
        after all statistical/appearance processing and immediately before ID
        allocation, so tracker fragmentation or a registry miss cannot create
        a nearby duplicate marker.

        The candidate is attached to the nearest existing marker for identity
        bookkeeping, but its geometry is not allowed to move the published
        landmark coordinate.
        """
        candidate_center = np.asarray(
            [fit['x'], fit['y']],
            dtype=np.float64,
        )

        nearest_landmark = None
        nearest_distance = float('inf')
        for landmark in self._landmarks.values():
            if landmark.class_id != int(candidate.class_id):
                continue

            landmark_center = landmark.center(
                self.ray_inlier_threshold,
                self.minimum_baseline,
            )
            distance = float(np.linalg.norm(
                candidate_center
                - np.asarray(
                    landmark_center,
                    dtype=np.float64,
                )
            ))
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_landmark = landmark

        if (
            nearest_landmark is not None
            and nearest_distance
            <= self.PUBLISH_SUPPRESSION_RADIUS_M
        ):
            # Keep the old published coordinate fixed.  Only identity and
            # appearance history are absorbed.
            self._absorb_candidate_geometry(
                candidate,
                nearest_landmark,
                fit,
                update_geometry=False,
            )
            self.logger.warning(
                f'YOLO 30cm publication guard: '
                f'candidate={candidate.candidate_id}, '
                f'drone={candidate.source_id}, '
                f'new_position=('
                f'{candidate_center[0]:.2f}, '
                f'{candidate_center[1]:.2f}), '
                f'suppressed_near_marker='
                f'{nearest_landmark.marker_id}, '
                f'distance={nearest_distance:.2f}m '
                f'<= {self.PUBLISH_SUPPRESSION_RADIUS_M:.2f}m'
            )
            self._remove_candidate(
                candidate.candidate_id)
            return None

        result = super()._create_landmark_from_candidate(
            candidate,
            fit,
        )
        if result is not None:
            self.logger.info(
                f'YOLO uncertainty commit: '
                f'marker={result["marker_id"]}, '
                f'radius95={fit.get("radius95", float("inf")):.2f}m, '
                f'condition={fit.get("condition_number", float("inf")):.1f}'
            )
        return result


# Use the uncertainty-aware implementation without changing launch or YAML.
RealtimeLandmarkRegistry = UncertaintyAwareLandmarkRegistry

class YoloBackend:
    """Projection and persistent landmark association for one source."""

    def __init__(
        self,
        source_id,
        detector,
        registry,
        camera_matrix,
        distortion,
        rotation_body_from_camera,
        target_height,
        maximum_ground_range,
        minimum_downward_ray,
    ):
        self.source_id = str(source_id)
        self.detector = detector
        self.registry = registry
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.rotation_body_from_camera = (
            rotation_body_from_camera)
        self.target_height = float(
            target_height)
        self.maximum_ground_range = float(
            maximum_ground_range)
        self.minimum_downward_ray = float(
            minimum_downward_ray)

    def process_tracked(
        self,
        tracked,
        drone_pose,
        image_shape=None,
    ):
        if drone_pose is None:
            return [], len(tracked)

        projected = []
        for detection in tracked:
            pixel_u, pixel_v = (
                detection['ground_px'])
            hit = pixel_ray_to_world(
                u=pixel_u,
                v=pixel_v,
                camera_matrix=self.camera_matrix,
                distortion=self.distortion,
                drone_pose=drone_pose,
                rotation_body_from_camera=(
                    self.rotation_body_from_camera),
                ground_z=self.target_height,
                minimum_downward_ray=(
                    self.minimum_downward_ray),
                maximum_ground_range=(
                    self.maximum_ground_range),
            )
            if hit is None:
                continue

            projected.append({
                'track_id': int(
                    detection['track_id']),
                'class_id': int(
                    detection['class_id']),
                'class_name': str(
                    detection['class_name']),
                'confidence': float(
                    detection['confidence']),
                'bbox': tuple(
                    detection['bbox']),
                'image_center': tuple(
                    detection['image_center']),
                'x': float(hit['x']),
                'y': float(hit['y']),
                'z': float(hit['z']),
                'ground_range': float(
                    hit['ground_range']),
                'ray_origin_x': float(
                    hit['ray_origin_x']),
                'ray_origin_y': float(
                    hit['ray_origin_y']),
                'ray_direction_x': float(
                    hit['ray_direction_x']),
                'ray_direction_y': float(
                    hit['ray_direction_y']),
                # Keep the DeepSORT-style descriptor.  v14 computed it in the
                # detector but accidentally dropped it at this boundary, which
                # made every registry log show appearance=inf.
                'appearance': detection.get('appearance'),
            })

        projection_context = None
        if image_shape is not None:
            image_height, image_width = image_shape[:2]
            projection_context = {
                'drone_pose': drone_pose,
                'camera_matrix': self.camera_matrix,
                'distortion': self.distortion,
                'rotation_body_from_camera': (
                    self.rotation_body_from_camera),
                'image_height': int(image_height),
                'image_width': int(image_width),
                'ground_z': self.target_height,
            }

        confirmed = self.registry.update_frame(
            source_id=self.source_id,
            detections=projected,
            now=time.monotonic(),
            projection_context=projection_context,
        )
        return confirmed, len(tracked)

    def process(self, frame_bgr, drone_pose):
        tracked = self.detector.track_raw(
            self.source_id,
            frame_bgr,
        )
        return self.process_tracked(
            tracked,
            drone_pose,
            image_shape=frame_bgr.shape[:2],
        )

    def draw_latest_overlay(
        self,
        frame_bgr,
    ):
        return self.detector.draw_latest_overlay(
            self.source_id,
            frame_bgr,
        )


class InferenceScheduler:
    """DeepStream-style latest-frame micro-batch worker.

    Each source owns one mailbox.  New RX frames overwrite old unprocessed
    frames.  The worker takes all currently available sources together, runs one
    shared detector batch when the ONNX model supports it, and never builds a
    latency-growing FIFO.
    """

    def __init__(self, node, source_order):
        self.node = node
        self.source_order = list(
            source_order)
        self._pending = {}
        self._condition = threading.Condition()
        self._maximum_frame_age = 0.75
        self._batch_wait_sec = 0.012
        self._last_metrics_log = 0.0
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name='aideck_batch_inference',
        )
        self._thread.start()

    def submit(
        self,
        link,
        frame,
        receive_time,
    ):
        with self._condition:
            # Latest frame wins.  This is intentional frame dropping, not a
            # queue.  It bounds end-to-end latency under GPU overload.
            self._pending[
                link.drone_id
            ] = (
                link,
                frame,
                float(receive_time),
            )
            self._condition.notify()

    def _take_batch(self):
        with self._condition:
            while (
                rclpy.ok()
                and not self._pending
            ):
                self._condition.wait(
                    timeout=0.5)
            if not rclpy.ok():
                return []

            deadline = (
                time.monotonic()
                + self._batch_wait_sec
            )
            while (
                len(self._pending)
                < len(self.source_order)
                and rclpy.ok()
            ):
                remaining = (
                    deadline
                    - time.monotonic()
                )
                if remaining <= 0.0:
                    break
                self._condition.wait(
                    timeout=remaining)

            items = []
            for source_id in self.source_order:
                item = self._pending.pop(
                    source_id,
                    None,
                )
                if item is not None:
                    items.append(item)

            if not items and self._pending:
                _source_id, item = (
                    self._pending.popitem())
                items.append(item)
            return items

    def _publish_results(
        self,
        link,
        tracked,
        receive_time,
        inference_end,
        image_shape,
    ):
        drone_pose, pose_gap = (
            link._pose_for_frame(
                receive_time)
        )
        confirmed, raw_count = (
            link.backend.process_tracked(
                tracked,
                drone_pose,
                image_shape=image_shape,
            )
        )

        if (
            raw_count > 0
            and drone_pose is None
        ):
            reason = (
                'no pose'
                if pose_gap is None
                else (
                    f'pose gap '
                    f'{pose_gap:.2f}s'
                )
            )
            link.node.get_logger().warn(
                f'{link.drone_id}: '
                f'target tracked but world '
                f'projection skipped ({reason})')

        for result in confirmed:
            message = MarkerDetection()
            message.header.frame_id = 'map'
            message.header.stamp = (
                link.node.get_clock()
                .now().to_msg()
            )
            message.drone_id = (
                link.drone_id)
            message.marker_id = int(
                result['marker_id'])
            message.position.x = float(
                result['x'])
            message.position.y = float(
                result['y'])
            message.position.z = float(
                result['z'])
            link.node.detections_pub.publish(
                message)

        link.note_inference(
            receive_time=receive_time,
            finish_time=inference_end,
            target_count=raw_count,
        )

    def _run_yolo_batch(self, items):
        valid_items = []
        now_wall = time.time()
        for item in items:
            link, frame, receive_time = item
            if (
                now_wall - receive_time
                <= self._maximum_frame_age
            ):
                valid_items.append(item)
            else:
                link.note_stale_drop()

        if not valid_items:
            return

        source_frames = {
            link.drone_id: frame
            for link, frame, _receive_time
            in valid_items
        }
        detector = (
            valid_items[0][0]
            .backend.detector
        )

        tracked_by_source, metrics = (
            detector.track_batch(
                source_frames)
        )
        inference_end = time.time()

        for link, frame, receive_time in valid_items:
            self._publish_results(
                link,
                tracked_by_source.get(
                    link.drone_id,
                    [],
                ),
                receive_time,
                inference_end,
                image_shape=frame.shape[:2],
            )

        now_mono = time.monotonic()
        if (
            now_mono - self._last_metrics_log
            >= 5.0
        ):
            self._last_metrics_log = now_mono
            self.node.get_logger().info(
                f'YOLO realtime batch: '
                f'sources={metrics["batch_size"]}, '
                f'batch_used='
                f'{metrics["batch_used"]}, '
                f'pre='
                f'{metrics["preprocess_ms"]:.1f}ms, '
                f'infer='
                f'{metrics["inference_ms"]:.1f}ms, '
                f'post='
                f'{metrics["postprocess_ms"]:.1f}ms')

    def _run(self):
        while rclpy.ok():
            items = self._take_batch()
            if not items:
                continue

            first_link = items[0][0]
            if isinstance(
                first_link.backend,
                YoloBackend,
            ):
                try:
                    self._run_yolo_batch(
                        items)
                except Exception as exception:
                    self.node.get_logger().error(
                        f'YOLO batch perception '
                        f'failed: {exception}')
                continue

            # ArUco remains a per-frame CPU path.
            for link, frame, receive_time in items:
                link.process_frame(
                    frame,
                    receive_time,
                )


class DroneLink:
    RADIO_STALE_SEC = 1.0

    def __init__(
        self,
        node,
        drone_id,
        wifi_ip,
        wifi_port,
        backend,
        bridge,
        scheduler,
        camera_latency_sec,
        pose_tolerance_sec,
    ):
        self.node = node
        self.drone_id = str(drone_id)
        self.wifi_ip = str(wifi_ip)
        self.wifi_port = int(wifi_port)
        self.backend = backend
        self.bridge = bridge
        self.scheduler = scheduler
        self.camera_latency_sec = max(
            0.0,
            float(camera_latency_sec),
        )
        self.pose_tolerance_sec = max(
            0.05,
            float(pose_tolerance_sec),
        )

        self._pose_lock = threading.Lock()
        self._pose_history = deque(
            maxlen=100)
        self.wifi_connected = False
        self.battery_voltage = 0.0

        # The user-facing topic stays exactly /cfX/image_raw.
        self.image_pub = node.create_publisher(
            Image,
            f'/{self.drone_id}/image_raw',
            2,
        )

        # RX, inference, and ROS image publication are three independent stages.
        # The display mailbox also stores only the newest frame.
        self._display_condition = (
            threading.Condition())
        self._display_frame = None
        self._display_sequence = 0

        self._metrics_lock = threading.Lock()
        self._rx_times = deque(maxlen=200)
        self._video_times = deque(maxlen=200)
        self._infer_times = deque(maxlen=200)
        self._infer_latency = deque(maxlen=200)
        self._stale_drops = 0

        self._receiver_thread = threading.Thread(
            target=self._receiver_loop,
            daemon=True,
            name=f'{self.drone_id}_aideck_rx',
        )
        self._display_thread = threading.Thread(
            target=self._display_loop,
            daemon=True,
            name=f'{self.drone_id}_image_raw_pub',
        )
        self._receiver_thread.start()
        self._display_thread.start()

    def update_pose(
        self,
        timestamp,
        position,
        quaternion,
    ):
        with self._pose_lock:
            self._pose_history.append(
                PoseSample(
                    timestamp,
                    position,
                    quaternion,
                )
            )

    def update_battery(self, voltage):
        self.battery_voltage = float(
            voltage)

    def radio_connected(self, now):
        with self._pose_lock:
            if not self._pose_history:
                return False
            timestamp = (
                self._pose_history[-1]
                .timestamp
            )
        return (
            float(now) - timestamp
            < self.RADIO_STALE_SEC
        )

    def _pose_for_frame(
        self,
        receive_time,
    ):
        target_time = (
            float(receive_time)
            - self.camera_latency_sec
        )
        with self._pose_lock:
            samples = list(
                self._pose_history)

        if not samples:
            return None, None

        if target_time <= samples[0].timestamp:
            gap = abs(
                target_time
                - samples[0].timestamp
            )
            if gap > self.pose_tolerance_sec:
                return None, gap
            sample = samples[0]
            return DronePose(
                *sample.position,
                sample.quaternion,
            ), gap

        if target_time >= samples[-1].timestamp:
            gap = abs(
                target_time
                - samples[-1].timestamp
            )
            if gap > self.pose_tolerance_sec:
                return None, gap
            sample = samples[-1]
            return DronePose(
                *sample.position,
                sample.quaternion,
            ), gap

        for first, second in zip(
            samples[:-1],
            samples[1:],
        ):
            if (
                first.timestamp
                <= target_time
                <= second.timestamp
            ):
                span = (
                    second.timestamp
                    - first.timestamp
                )
                ratio = (
                    0.0
                    if span <= 1.0e-9
                    else (
                        target_time
                        - first.timestamp
                    ) / span
                )
                position = (
                    first.position
                    + ratio
                    * (
                        second.position
                        - first.position
                    )
                )
                quaternion = quaternion_slerp(
                    first.quaternion,
                    second.quaternion,
                    ratio,
                )
                gap = min(
                    target_time
                    - first.timestamp,
                    second.timestamp
                    - target_time,
                )
                return DronePose(
                    *position,
                    quaternion,
                ), gap

        return None, None

    def _submit_display(
        self,
        frame,
        receive_time,
    ):
        with self._display_condition:
            self._display_sequence += 1
            self._display_frame = (
                self._display_sequence,
                frame,
                float(receive_time),
            )
            self._display_condition.notify()

    def _display_loop(self):
        last_sequence = 0
        while rclpy.ok():
            with self._display_condition:
                while (
                    rclpy.ok()
                    and (
                        self._display_frame is None
                        or self._display_frame[0]
                        == last_sequence
                    )
                ):
                    self._display_condition.wait(
                        timeout=0.5)
                if not rclpy.ok():
                    return
                sequence, source_frame, receive_time = (
                    self._display_frame)
                last_sequence = sequence

            frame = source_frame.copy()
            if isinstance(
                self.backend,
                YoloBackend,
            ):
                self.backend.draw_latest_overlay(
                    frame)

            try:
                image_message = (
                    self.bridge.cv2_to_imgmsg(
                        frame,
                        encoding='bgr8',
                    )
                )
                image_message.header.frame_id = (
                    self.drone_id)
                image_message.header.stamp = (
                    self.node.get_clock()
                    .now().to_msg()
                )
                self.image_pub.publish(
                    image_message)
                with self._metrics_lock:
                    self._video_times.append(
                        time.monotonic())
            except Exception as exception:
                self.node.get_logger().warn(
                    f'{self.drone_id} '
                    f'image_raw publish failed: '
                    f'{exception}')

    def _receiver_loop(self):
        while rclpy.ok():
            sock = try_connect(
                self.wifi_ip,
                self.wifi_port,
            )
            if sock is None:
                self.wifi_connected = False
                time.sleep(2.0)
                continue

            self.wifi_connected = True
            self.node.get_logger().info(
                f'{self.drone_id} WiFi '
                f'connected to '
                f'{self.wifi_ip}:'
                f'{self.wifi_port}')

            frame_count = 0
            try:
                while rclpy.ok():
                    frame = receive_frame(
                        sock,
                        logger=(
                            self.node.get_logger()),
                    )
                    if frame is None:
                        continue

                    frame_count += 1
                    receive_time = time.time()
                    with self._metrics_lock:
                        self._rx_times.append(
                            time.monotonic())

                    if frame_count == 1:
                        self.node.get_logger().info(
                            f'{self.drone_id} '
                            f'first frame '
                            f'{frame.shape[1]}x'
                            f'{frame.shape[0]}')
                    elif frame_count % 100 == 0:
                        self.node.get_logger().info(
                            f'{self.drone_id}: '
                            f'{frame_count} '
                            f'frames received')

                    # Neither ROS serialization nor GPU inference blocks TCP RX.
                    self._submit_display(
                        frame,
                        receive_time,
                    )
                    self.scheduler.submit(
                        self,
                        frame,
                        receive_time,
                    )

            except socket.timeout:
                self.node.get_logger().warn(
                    f'{self.drone_id}: '
                    f'AI-deck connected but '
                    f'no complete frame arrived '
                    f'for 10s')
            except Exception as exception:
                self.node.get_logger().warn(
                    f'{self.drone_id} '
                    f'video receiver failed: '
                    f'{exception}')
            finally:
                self.wifi_connected = False
                try:
                    sock.close()
                except OSError:
                    pass

            time.sleep(1.0)

    def process_frame(
        self,
        frame,
        receive_time,
    ):
        """Compatibility path for ArUco."""
        drone_pose, pose_gap = (
            self._pose_for_frame(
                receive_time)
        )
        try:
            results, raw_count = (
                self.backend.process(
                    frame,
                    drone_pose,
                )
            )
        except Exception as exception:
            self.node.get_logger().error(
                f'{self.drone_id} '
                f'perception failed: '
                f'{exception}')
            results = []
            raw_count = 0

        if (
            raw_count > 0
            and drone_pose is None
        ):
            reason = (
                'no pose'
                if pose_gap is None
                else (
                    f'pose gap '
                    f'{pose_gap:.2f}s'
                )
            )
            self.node.get_logger().warn(
                f'{self.drone_id}: '
                f'target tracked but '
                f'world projection skipped '
                f'({reason})')

        for result in results:
            message = MarkerDetection()
            message.header.frame_id = 'map'
            message.header.stamp = (
                self.node.get_clock()
                .now().to_msg()
            )
            message.drone_id = (
                self.drone_id)
            message.marker_id = int(
                result['marker_id'])
            message.position.x = float(
                result['x'])
            message.position.y = float(
                result['y'])
            message.position.z = float(
                result['z'])
            self.node.detections_pub.publish(
                message)

    def note_inference(
        self,
        receive_time,
        finish_time,
        target_count,
    ):
        del target_count
        with self._metrics_lock:
            self._infer_times.append(
                time.monotonic())
            self._infer_latency.append(
                max(
                    0.0,
                    float(finish_time)
                    - float(receive_time),
                )
            )

    def note_stale_drop(self):
        with self._metrics_lock:
            self._stale_drops += 1

    @staticmethod
    def _rate(times):
        if len(times) < 2:
            return 0.0
        span = float(
            times[-1] - times[0])
        if span <= 1.0e-9:
            return 0.0
        return float(
            len(times) - 1
        ) / span

    def performance_summary(self):
        with self._metrics_lock:
            rx_times = list(
                self._rx_times)
            video_times = list(
                self._video_times)
            infer_times = list(
                self._infer_times)
            latencies = list(
                self._infer_latency)
            stale_drops = int(
                self._stale_drops)

        p95_latency = (
            float(np.percentile(
                latencies, 95.0))
            if latencies
            else 0.0
        )
        return (
            f'{self.drone_id}: '
            f'rx={self._rate(rx_times):.1f}Hz, '
            f'image_raw='
            f'{self._rate(video_times):.1f}Hz, '
            f'infer={self._rate(infer_times):.1f}Hz, '
            f'infer_p95='
            f'{p95_latency * 1000.0:.0f}ms, '
            f'stale_drops={stale_drops}'
        )

class RealPerceptionNode(Node):
    def __init__(self):
        super().__init__('real_perception_node')

        self.declare_parameter(
            'drone_ids', ['cf6', 'cf7'])
        self.declare_parameter(
            'wifi_ips', [''])
        self.declare_parameter(
            'wifi_port', 5000)
        self.declare_parameter(
            'marker_size', 0.14)
        self.declare_parameter(
            'camera_intrinsics_path', '')
        self.declare_parameter(
            'camera_pitch_degs', [45.0])
        self.declare_parameter(
            'camera_latency_sec', 0.0)
        self.declare_parameter(
            'pose_tolerance_sec', 0.30)
        self.declare_parameter(
            'detection_backend', 'yolo')
        self.declare_parameter(
            'yolo_weights_path', '')

        self.declare_parameter(
            'yolo_confidence_threshold', 0.35)
        self.declare_parameter(
            'yolo_low_confidence_threshold', 0.10)
        self.declare_parameter(
            'yolo_nms_threshold', 0.45)
        self.declare_parameter(
            'yolo_image_size', 416)
        self.declare_parameter(
            'yolo_maximum_detections', 20)
        self.declare_parameter(
            'yolo_force_grayscale', True)
        self.declare_parameter(
            'yolo_use_clahe', False)
        self.declare_parameter(
            'yolo_track_buffer', 240)
        self.declare_parameter(
            'yolo_match_threshold', 0.90)
        self.declare_parameter(
            'yolo_new_track_threshold', 0.50)

        self.declare_parameter(
            'yolo_target_height', 0.0)
        self.declare_parameter(
            'yolo_max_ground_range', 3.0)
        self.declare_parameter(
            'yolo_min_downward_ray', 0.08)

        self.declare_parameter(
            'registry_sample_window', 40)
        self.declare_parameter(
            'registry_confirmation_hits', 10)
        self.declare_parameter(
            'registry_minimum_confirmation_age', 1.0)
        self.declare_parameter(
            'registry_maximum_ray_residual', 0.22)
        self.declare_parameter(
            'registry_ray_inlier_threshold', 0.24)
        self.declare_parameter(
            'registry_ray_match_threshold', 0.30)
        self.declare_parameter(
            'registry_minimum_parallax_deg', 5.0)
        self.declare_parameter(
            'registry_minimum_baseline', 0.30)
        self.declare_parameter(
            'registry_minimum_triangulation_inliers', 7)
        self.declare_parameter(
            'registry_covariance_floor', 0.20)
        self.declare_parameter(
            'registry_association_chi2_gate', 16.0)
        self.declare_parameter(
            'registry_association_max_distance', 0.60)
        self.declare_parameter(
            'registry_ambiguity_chi2_gate', 36.0)
        self.declare_parameter(
            'registry_ambiguity_max_distance', 1.20)
        self.declare_parameter(
            'registry_duplicate_minimum_bbox_iou', 0.15)
        self.declare_parameter(
            'registry_duplicate_maximum_pixel_distance', 24.0)
        self.declare_parameter(
            'registry_distinct_evidence_frames', 5)
        self.declare_parameter(
            'registry_distinct_maximum_bbox_iou', 0.05)
        self.declare_parameter(
            'registry_distinct_minimum_pixel_distance', 35.0)

        self.declare_parameter(
            'registry_bundle_inlier_threshold', 0.30)
        self.declare_parameter(
            'registry_bundle_minimum_group_inlier_ratio', 0.55)
        self.declare_parameter(
            'registry_bundle_maximum_group_median_error', 0.30)
        self.declare_parameter(
            'registry_appearance_merge_threshold', 0.12)
        self.declare_parameter(
            'registry_appearance_max_distance', 1.20)
        self.declare_parameter(
            'registry_hypothesis_spatial_gate', 0.45)
        self.declare_parameter(
            'registry_hypothesis_minimum_tracklets', 3)
        self.declare_parameter(
            'registry_hypothesis_minimum_separation', 0.55)
        self.declare_parameter(
            'registry_hypothesis_separation_chi2', 25.0)
        self.declare_parameter(
            'registry_hypothesis_timeout', 45.0)
        self.declare_parameter(
            'registry_tracklet_timeout', 30.0)

        self.drone_ids = list(
            self.get_parameter('drone_ids').value)
        wifi_ips = list(
            self.get_parameter('wifi_ips').value)
        if len(wifi_ips) != len(self.drone_ids):
            raise RuntimeError(
                'wifi_ips must match drone_ids')

        pitch_values = list(
            self.get_parameter(
                'camera_pitch_degs').value)
        if len(pitch_values) == 1:
            pitch_values *= len(self.drone_ids)
        if len(pitch_values) != len(self.drone_ids):
            raise RuntimeError(
                'camera_pitch_degs must contain one '
                'value or one per drone')

        camera_matrix, distortion = (
            self._load_intrinsics(
                str(self.get_parameter(
                    'camera_intrinsics_path').value)
            )
        )

        backend_name = str(
            self.get_parameter(
                'detection_backend').value)
        marker_size = float(
            self.get_parameter('marker_size').value)

        shared_detector = None
        registry = None
        if backend_name == 'yolo':
            weights_path = str(
                self.get_parameter(
                    'yolo_weights_path').value)
            if not weights_path:
                raise RuntimeError(
                    'yolo_weights_path is empty')

            shared_detector = YoloDetector(
                weights_path=weights_path,
                confidence_threshold=(
                    self.get_parameter(
                        'yolo_confidence_threshold').value),
                low_confidence_threshold=(
                    self.get_parameter(
                        'yolo_low_confidence_threshold').value),
                nms_threshold=(
                    self.get_parameter(
                        'yolo_nms_threshold').value),
                image_size=(
                    self.get_parameter(
                        'yolo_image_size').value),
                maximum_detections=(
                    self.get_parameter(
                        'yolo_maximum_detections').value),
                force_grayscale=(
                    self.get_parameter(
                        'yolo_force_grayscale').value),
                use_clahe=(
                    self.get_parameter(
                        'yolo_use_clahe').value),
                track_buffer=(
                    self.get_parameter(
                        'yolo_track_buffer').value),
                match_threshold=(
                    self.get_parameter(
                        'yolo_match_threshold').value),
                new_track_threshold=(
                    self.get_parameter(
                        'yolo_new_track_threshold').value),
                source_ids=self.drone_ids,
            )

            registry = RealtimeLandmarkRegistry(
                logger=self.get_logger(),
                marker_size=marker_size,
                sample_window=(
                    self.get_parameter(
                        'registry_sample_window').value),
                confirmation_hits=(
                    self.get_parameter(
                        'registry_confirmation_hits').value),
                minimum_confirmation_age=(
                    self.get_parameter(
                        'registry_minimum_confirmation_age').value),
                maximum_ray_residual=(
                    self.get_parameter(
                        'registry_maximum_ray_residual').value),
                ray_inlier_threshold=(
                    self.get_parameter(
                        'registry_ray_inlier_threshold').value),
                ray_match_threshold=(
                    self.get_parameter(
                        'registry_ray_match_threshold').value),
                minimum_parallax_deg=(
                    self.get_parameter(
                        'registry_minimum_parallax_deg').value),
                minimum_baseline=(
                    self.get_parameter(
                        'registry_minimum_baseline').value),
                minimum_triangulation_inliers=(
                    self.get_parameter(
                        'registry_minimum_triangulation_inliers').value),
                covariance_floor=(
                    self.get_parameter(
                        'registry_covariance_floor').value),
                association_chi2_gate=(
                    self.get_parameter(
                        'registry_association_chi2_gate').value),
                association_max_distance=(
                    self.get_parameter(
                        'registry_association_max_distance').value),
                ambiguity_chi2_gate=(
                    self.get_parameter(
                        'registry_ambiguity_chi2_gate').value),
                ambiguity_max_distance=(
                    self.get_parameter(
                        'registry_ambiguity_max_distance').value),
                duplicate_minimum_bbox_iou=(
                    self.get_parameter(
                        'registry_duplicate_minimum_bbox_iou').value),
                duplicate_maximum_pixel_distance=(
                    self.get_parameter(
                        'registry_duplicate_maximum_pixel_distance').value),
                distinct_evidence_frames=(
                    self.get_parameter(
                        'registry_distinct_evidence_frames').value),
                distinct_maximum_bbox_iou=(
                    self.get_parameter(
                        'registry_distinct_maximum_bbox_iou').value),
                distinct_minimum_pixel_distance=(
                    self.get_parameter(
                        'registry_distinct_minimum_pixel_distance').value),
                bundle_inlier_threshold=(
                    self.get_parameter(
                        'registry_bundle_inlier_threshold').value),
                bundle_minimum_group_inlier_ratio=(
                    self.get_parameter(
                        'registry_bundle_minimum_group_inlier_ratio').value),
                bundle_maximum_group_median_error=(
                    self.get_parameter(
                        'registry_bundle_maximum_group_median_error').value),
                appearance_merge_threshold=(
                    self.get_parameter(
                        'registry_appearance_merge_threshold').value),
                appearance_max_distance=(
                    self.get_parameter(
                        'registry_appearance_max_distance').value),
                hypothesis_spatial_gate=(
                    self.get_parameter(
                        'registry_hypothesis_spatial_gate').value),
                hypothesis_minimum_tracklets=(
                    self.get_parameter(
                        'registry_hypothesis_minimum_tracklets').value),
                hypothesis_minimum_separation=(
                    self.get_parameter(
                        'registry_hypothesis_minimum_separation').value),
                hypothesis_separation_chi2=(
                    self.get_parameter(
                        'registry_hypothesis_separation_chi2').value),
                hypothesis_timeout=(
                    self.get_parameter(
                        'registry_hypothesis_timeout').value),
                tracklet_timeout=(
                    self.get_parameter(
                        'registry_tracklet_timeout').value),
            )

        self.states_pub = self.create_publisher(
            DroneState, '/states', 20)
        self.detections_pub = self.create_publisher(
            MarkerDetection, '/detections', 10)
        self.link_status_pub = self.create_publisher(
            LinkStatusArray,
            '/mission/link_status',
            10,
        )

        self.scheduler = InferenceScheduler(
            self, self.drone_ids)
        bridge = CvBridge()
        self.links = {}
        self.pose_subscriptions = []
        self.status_subscriptions = []

        for drone_id, wifi_ip, pitch in zip(
            self.drone_ids,
            wifi_ips,
            pitch_values,
        ):
            rotation = camera_to_body_rotation(pitch)
            if backend_name == 'yolo':
                backend = YoloBackend(
                    source_id=drone_id,
                    detector=shared_detector,
                    registry=registry,
                    camera_matrix=camera_matrix,
                    distortion=distortion,
                    rotation_body_from_camera=rotation,
                    target_height=(
                        self.get_parameter(
                            'yolo_target_height').value),
                    maximum_ground_range=(
                        self.get_parameter(
                            'yolo_max_ground_range').value),
                    minimum_downward_ray=(
                        self.get_parameter(
                            'yolo_min_downward_ray').value),
                )
            elif backend_name == 'aruco':
                backend = ArucoBackend(
                    camera_matrix=camera_matrix,
                    distortion=distortion,
                    marker_size_m=marker_size,
                    rotation_body_from_camera=rotation,
                )
            else:
                raise RuntimeError(
                    f'unknown detection backend '
                    f'{backend_name}')

            link = DroneLink(
                node=self,
                drone_id=drone_id,
                wifi_ip=wifi_ip,
                wifi_port=int(
                    self.get_parameter(
                        'wifi_port').value),
                backend=backend,
                bridge=bridge,
                scheduler=self.scheduler,
                camera_latency_sec=(
                    self.get_parameter(
                        'camera_latency_sec').value),
                pose_tolerance_sec=(
                    self.get_parameter(
                        'pose_tolerance_sec').value),
            )
            self.links[drone_id] = link

            self.pose_subscriptions.append(
                self.create_subscription(
                    PoseStamped,
                    f'/{drone_id}/pose',
                    self._make_pose_callback(
                        drone_id),
                    20,
                )
            )
            self.status_subscriptions.append(
                self.create_subscription(
                    Status,
                    f'/{drone_id}/status',
                    self._make_status_callback(
                        drone_id),
                    10,
                )
            )

        self.create_timer(
            0.5, self._publish_link_status)
        self.create_timer(
            5.0, self._publish_performance)
        self.get_logger().info(
            f'real_perception_node backend={backend_name}, '
            f'drones={self.drone_ids}, '
            f'pitches={pitch_values}')
        if backend_name == 'yolo':
            self.get_logger().info(
                'YOLO realtime pipeline: one shared ORT CUDA session, '
                'per-drone ByteTrack, latest-frame micro-batching, '
                'uncertainty-fusion + hard 30cm publish guard, image_raw only')
            self.get_logger().info(
                f'YOLO runtime providers='
                f'{shared_detector.active_providers}, '
                f'input={shared_detector.input_width}x'
                f'{shared_detector.input_height}, '
                f'dynamic_batch='
                f'{shared_detector.dynamic_batch}')

    def _publish_performance(self):
        for link in self.links.values():
            self.get_logger().info(
                link.performance_summary())

    def _load_intrinsics(self, path):
        if not path:
            self.get_logger().warn(
                'camera intrinsics missing; using '
                'placeholder 324x244 values')
            return (
                np.array(
                    [
                        [320.0, 0.0, 162.0],
                        [0.0, 320.0, 122.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                np.zeros((5, 1), dtype=np.float32),
            )

        with open(
            path, 'r', encoding='utf-8'
        ) as file_handle:
            data = yaml.safe_load(file_handle)

        camera_matrix = np.asarray(
            data['camera_matrix'],
            dtype=np.float32,
        )
        distortion = np.asarray(
            data['dist_coeffs'],
            dtype=np.float32,
        )

        placeholder = np.array(
            [
                [320.0, 0.0, 162.0],
                [0.0, 320.0, 122.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        if (
            np.allclose(camera_matrix, placeholder)
            and np.allclose(distortion, 0.0)
        ):
            self.get_logger().warn(
                'camera intrinsics are still placeholder; '
                'marker identity is protected by tracking, '
                'but absolute coordinates remain approximate')

        return camera_matrix, distortion

    def _make_pose_callback(self, drone_id):
        def callback(message):
            timestamp = (
                float(message.header.stamp.sec)
                + float(message.header.stamp.nanosec)
                * 1.0e-9
            )
            if timestamp <= 0.0:
                timestamp = time.time()

            position = (
                float(message.pose.position.x),
                float(message.pose.position.y),
                float(message.pose.position.z),
            )
            quaternion = (
                float(message.pose.orientation.x),
                float(message.pose.orientation.y),
                float(message.pose.orientation.z),
                float(message.pose.orientation.w),
            )
            self.links[drone_id].update_pose(
                timestamp,
                position,
                quaternion,
            )

            state = DroneState()
            state.header = message.header
            if not state.header.frame_id:
                state.header.frame_id = 'map'
            state.drone_id = drone_id
            state.position.x = position[0]
            state.position.y = position[1]
            state.position.z = position[2]
            state.yaw = quaternion_to_yaw(
                message.pose.orientation)
            self.states_pub.publish(state)

        return callback

    def _make_status_callback(self, drone_id):
        def callback(message):
            self.links[drone_id].update_battery(
                message.battery_voltage)
        return callback

    def _publish_link_status(self):
        now = time.time()
        array = LinkStatusArray()
        for drone_id, link in self.links.items():
            array.status.append(
                LinkStatus(
                    drone_id=drone_id,
                    radio_connected=(
                        link.radio_connected(now)),
                    wifi_connected=(
                        link.wifi_connected),
                    battery_voltage=(
                        link.battery_voltage),
                )
            )
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