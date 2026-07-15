#!/usr/bin/env python3
"""Mission state machine using onboard Crazyflie trajectories.

The coverage path is planned on the Jetson, converted once into degree-7
polynomial pieces, uploaded to each Crazyflie, and executed by the onboard
high-level commander. No low-level /cmd_full_state stream is used during
coverage, so YOLO/GCS/CPU load cannot starve the flight setpoint stream.

The node keeps the existing ROS mission topics and GCS services. It also:
- compresses collinear grid-cell waypoints before trajectory generation;
- uses a seventh-order smoothstep with zero velocity/acceleration/jerk at
  segment endpoints;
- monitors measured flight state and commands a safety land when a drone
  repeatedly leaves the configured flight envelope;
- performs a final same-drone spatial duplicate guard on marker reports.
"""
import math
import time

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from builtin_interfaces.msg import Duration as DurationMsg
from crazyflie_interfaces.msg import TrajectoryPolynomialPiece
from crazyflie_interfaces.srv import (
    GoTo,
    Land,
    StartTrajectory,
    Takeoff,
    UploadTrajectory,
)
from geometry_msgs.msg import Point, Point32
from geometry_msgs.msg import Polygon as PolygonMsg
from geometry_msgs.msg import PoseStamped
from mission_interfaces.msg import (
    CoveragePath, CoveragePathArray, DroneProgress, DroneProgressArray,
    MarkerDetection, MarkerRecord, MarkerRecordArray, ZoneAssignment, ZoneAssignmentArray,
)
from nav_msgs.msg import Path
from std_msgs.msg import String
from std_srvs.srv import Empty, Trigger

from mission_control.mission_planner import plan_zones

LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
)


def seconds_to_duration_msg(seconds):
    seconds = max(0.0, seconds)
    d = DurationMsg()
    d.sec = int(seconds)
    d.nanosec = int((seconds - int(seconds)) * 1e9)
    return d


def dist2d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class DroneHandle:
    """Per-drone high-level commander clients and trajectory state."""

    TRAJECTORY_ID = 1

    def __init__(self, node, drone_id, home_position):
        self.node = node
        self.drone_id = str(drone_id)
        self.home_position = tuple(home_position)
        prefix = '/' + self.drone_id

        self.takeoff_client = node.create_client(
            Takeoff, prefix + '/takeoff')
        self.land_client = node.create_client(
            Land, prefix + '/land')
        self.go_to_client = node.create_client(
            GoTo, prefix + '/go_to')
        self.upload_trajectory_client = node.create_client(
            UploadTrajectory, prefix + '/upload_trajectory')
        self.start_trajectory_client = node.create_client(
            StartTrajectory, prefix + '/start_trajectory')
        self.emergency_client = node.create_client(
            Empty, prefix + '/emergency')

        # Original cell-center path used for GCS progress.
        self.waypoints = []

        # Collinearity-compressed path used for onboard trajectory execution.
        self.flight_waypoints = []
        self.trajectory_pieces = []
        self.trajectory_duration = 0.0
        self.upload_future = None
        self.start_future = None
        self.covering_start_time = None

        self.visited_mask = set()
        self.arrived_index = 0
        self.done = False
        self.last_target_xy = (
            float(home_position[0]),
            float(home_position[1]),
        )

    def wait_for_services(self, node, timeout_sec):
        for client, name in (
            (self.takeoff_client, 'takeoff'),
            (self.land_client, 'land'),
            (self.go_to_client, 'go_to'),
            (self.upload_trajectory_client, 'upload_trajectory'),
            (self.start_trajectory_client, 'start_trajectory'),
        ):
            if not client.wait_for_service(timeout_sec=timeout_sec):
                raise RuntimeError(
                    f'{self.drone_id}/{name} service not available after '
                    f'{timeout_sec}s (is crazyflie_server running?)')

    def send_takeoff(self, height, duration):
        request = Takeoff.Request()
        request.group_mask = 0
        request.height = float(height)
        request.duration = seconds_to_duration_msg(duration)
        return self.takeoff_client.call_async(request)

    def send_land(self, height, duration):
        request = Land.Request()
        request.group_mask = 0
        request.height = float(height)
        request.duration = seconds_to_duration_msg(duration)
        return self.land_client.call_async(request)

    def send_go_to(self, xyz, yaw_rad, duration):
        request = GoTo.Request()
        request.group_mask = 0
        request.relative = False
        request.goal = Point(
            x=float(xyz[0]),
            y=float(xyz[1]),
            z=float(xyz[2]),
        )
        request.yaw = float(yaw_rad)
        request.duration = seconds_to_duration_msg(duration)
        self.last_target_xy = (float(xyz[0]), float(xyz[1]))
        return self.go_to_client.call_async(request)

    def upload_trajectory(self):
        request = UploadTrajectory.Request()
        request.trajectory_id = self.TRAJECTORY_ID
        request.piece_offset = 0
        request.pieces = list(self.trajectory_pieces)
        self.upload_future = self.upload_trajectory_client.call_async(
            request)
        return self.upload_future

    def start_trajectory(self):
        request = StartTrajectory.Request()
        request.group_mask = 0
        request.trajectory_id = self.TRAJECTORY_ID
        request.timescale = 1.0
        request.reversed = False
        request.relative = False
        self.start_future = self.start_trajectory_client.call_async(
            request)
        return self.start_future

    def send_emergency(self):
        self.emergency_client.call_async(Empty.Request())


class ControlNode(Node):

    def __init__(self):
        super().__init__('control_node')

        self.declare_parameter('mission_map_path', '')
        # crazyswarm2's crazyflies.yaml (its `enabled` robots) is the single
        # source of truth for which drones fly -- sim.launch.py/real.launch.py
        # read it and pass the enabled id list in as this parameter.
        self.declare_parameter('drone_ids', ['cf6'])
        self.declare_parameter('min_leg_duration', 1.5)
        self.declare_parameter('leg_settle_margin', 0.5)
        self.declare_parameter('takeoff_duration', 2.0)
        self.declare_parameter('takeoff_settle_time', 2.5)
        self.declare_parameter('land_duration', 2.5)
        self.declare_parameter('land_settle_time', 3.0)
        self.declare_parameter('start_immediately', False)
        self.declare_parameter('dead_zone_margin', 0.15)
        self.declare_parameter('arrival_radius', 0.25)
        self.declare_parameter('trajectory_settle_time', 1.0)
        self.declare_parameter('safety_boundary_margin', 0.75)
        self.declare_parameter('safety_violation_ticks', 3)

        self.drone_ids = list(self.get_parameter('drone_ids').value)
        self.min_leg_duration = self.get_parameter('min_leg_duration').value
        self.leg_settle_margin = self.get_parameter('leg_settle_margin').value
        self.takeoff_duration = self.get_parameter('takeoff_duration').value
        self.takeoff_settle_time = self.get_parameter('takeoff_settle_time').value
        self.land_duration = self.get_parameter('land_duration').value
        self.land_settle_time = self.get_parameter('land_settle_time').value
        self.arrival_radius = self.get_parameter('arrival_radius').value
        self.dead_zone_margin = self.get_parameter('dead_zone_margin').value
        self.trajectory_settle_time = float(
            self.get_parameter('trajectory_settle_time').value)
        self.safety_boundary_margin = float(
            self.get_parameter('safety_boundary_margin').value)
        self.safety_violation_ticks = max(
            1, int(self.get_parameter('safety_violation_ticks').value))

        mission_map_path = self.get_parameter('mission_map_path').value
        if not mission_map_path:
            raise RuntimeError(
                'mission_control requires the mission_map_path parameter '
                '(path to mission_map.yaml)')
        self.mission_map = self._load_mission_map(mission_map_path)
        self.trajectory_settle_time = float(
            self.mission_map.get(
                'trajectory_settle_time',
                self.trajectory_settle_time,
            )
        )
        self.safety_boundary_margin = float(
            self.mission_map.get(
                'safety_boundary_margin',
                self.safety_boundary_margin,
            )
        )
        self.cruise_altitude = float(self.mission_map['uav_cruise_altitude'])
        self.coverage_line_spacing = float(self.mission_map['coverage_line_spacing'])
        # Desired peak speed for the onboard polynomial trajectory.
        # The segment duration accounts for the 7th-order smoothstep's
        # derivative peak, so the actual peak speed stays near this value.
        self.cruise_speed = max(
            0.05, float(self.mission_map.get('cruise_speed', 0.20)))
        self.safety_max_altitude = float(
            self.mission_map.get(
                'safety_max_altitude',
                self.cruise_altitude + 0.60,
            )
        )

        self.drones = {}
        # home_position here is just a startup placeholder -- _do_plan() below
        # overwrites it with the actual first waypoint of whatever zone this
        # drone ends up assigned, once that's known (see mission_planner.plan_zones).
        for drone_id in self.drone_ids:
            handle = DroneHandle(self, drone_id, [0.0, 0.0, 0.0])
            handle.wait_for_services(self, timeout_sec=5.0)
            self.drones[drone_id] = handle

        self.zones_pub = self.create_publisher(
            ZoneAssignmentArray, '/mission/zones', LATCHED_QOS)
        self.paths_pub = self.create_publisher(
            CoveragePathArray, '/mission/coverage_paths', LATCHED_QOS)
        self.markers_pub = self.create_publisher(
            MarkerRecordArray, '/mission/markers', LATCHED_QOS)
        # Latched: state transitions (PREPARE -> AWAITING_START -> ...)
        # are each published exactly once on entry (_set_state), not repeated
        # on a timer -- a plain non-latched publisher means any GCS that
        # (re)connects after a transition already happened just never sees it
        # and sits on the SharedState default ('UNKNOWN') forever, even though
        # control_node's own log clearly shows it reached e.g. AWAITING_START.
        self.state_pub = self.create_publisher(String, '/mission/state', LATCHED_QOS)
        # Authoritative "how far along its path is each drone" feed for the GCS --
        # this used to be *guessed* client-side from nearest-waypoint distance,
        # which is unreliable on a zig-zag path (many waypoints can be spatially
        # close to the current position without actually being "next"). control_node
        # already tracks the real wp_index itself, so it's simplest to just publish it.
        self.progress_pub = self.create_publisher(DroneProgressArray, '/mission/progress', 10)

        self.detections_sub = self.create_subscription(
            MarkerDetection, '/detections', self._on_detection, 10)
        self.detected_markers = {}  # canonical marker_id -> (x, y, z)
        self.marker_sources = {}    # canonical marker_id -> drone_id

        # Real measured position per drone (from /states, same feed cf_perception
        # publishes for the GCS). Used instead of blindly assuming a drone has
        # already reached the last commanded target -- if sim/real timing runs
        # faster or slower than our wall-clock leg-duration estimate, computing
        # the next leg's distance from a *wrong* assumed start point produces a
        # too-short duration for the real distance still left to cover, which
        # crazyswarm2's own go_to docs warn drives an unstable/runaway trajectory.
        # Flight monitoring subscribes directly to crazyflie_server pose
        # topics. It does not depend on the Docker perception process.
        self.live_xy = {}   # drone_id -> (x, y)
        self.live_xyz = {}  # drone_id -> (x, y, z)
        self.live_state_time = {}  # drone_id -> monotonic timestamp
        self._safety_violations = {
            drone_id: 0 for drone_id in self.drone_ids}
        self.pose_subscriptions = []
        for drone_id in self.drone_ids:
            self.pose_subscriptions.append(
                self.create_subscription(
                    PoseStamped,
                    f'/{drone_id}/pose',
                    self._make_pose_callback(drone_id),
                    20,
                )
            )

        self._start_requested = self.get_parameter('start_immediately').value
        self.start_srv = self.create_service(Trigger, '/mission/start', self._on_start_request)

        # Emergency kill switch (GCS button / 'k' key). Once tripped it cuts all
        # motors and permanently freezes the FSM -- there is no un-kill; recover
        # by restarting the mission stack.
        self._killed = False
        self.kill_srv = self.create_service(Trigger, '/mission/kill', self._on_kill_request)

        self.state = 'PREPARE'
        self._phase_deadline = None
        self._takeoff_sent = False
        self._align_sent = False
        self._trajectory_started = False
        self._trajectory_start_sent = False
        self._trajectory_start_request_time = None
        self._upload_deadline = None
        self._land_sent = False
        self._safety_land_sent = False
        self._upload_started = False

        self.timer = self.create_timer(0.1, self._tick)
        self.get_logger().info('control_node started, state=PREPARE')

    # ---- setup -----------------------------------------------------
    def _load_mission_map(self, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return data

    def _set_state(self, new_state):
        self.get_logger().info(f'mission state: {self.state} -> {new_state}')
        self.state = new_state
        self.state_pub.publish(String(data=new_state))

    # ---- callbacks ---------------------------------------------------
    def _make_pose_callback(self, drone_id):
        def callback(message):
            xyz = (
                float(message.pose.position.x),
                float(message.pose.position.y),
                float(message.pose.position.z),
            )
            self.live_xy[drone_id] = xyz[:2]
            self.live_xyz[drone_id] = xyz
            self.live_state_time[drone_id] = (
                time.monotonic())
        return callback

    def _on_detection(self, msg):
        """Store canonical marker identities from the perception registry.

        Mission control does not apply a spatial radius, per-drone quota, or
        object-count assumption. Arbitrary nearby objects remain independent.
        """
        marker_id = int(msg.marker_id)
        position = (
            float(msg.position.x),
            float(msg.position.y),
            float(msg.position.z),
        )
        source = str(msg.drone_id)

        if marker_id in self.detected_markers:
            # A repeated report for the same canonical ID is an update, not a
            # new target.
            self.detected_markers[marker_id] = position
            self.marker_sources[marker_id] = source
            self._publish_markers()
            return

        self.detected_markers[marker_id] = position
        self.marker_sources[marker_id] = source
        self.get_logger().info(
            f'marker {marker_id} detected by {source} at '
            f'({position[0]:.2f}, {position[1]:.2f})')
        self._publish_markers()

    def _on_start_request(self, request, response):
        self._start_requested = True
        response.success = True
        response.message = 'mission start requested'
        return response

    def _on_kill_request(self, request, response):
        # Cut motors on every drone immediately, then freeze the FSM (the
        # _tick early-return below) so no further setpoints/go_to commands go
        # out. Hard emergency stop: the drones drop rather than land -- that's
        # deliberate, a controlled drop beats driving into a wall.
        self._killed = True
        for handle in self.drones.values():
            handle.send_emergency()
        self._set_state('KILLED')
        self.get_logger().warn('EMERGENCY KILL: motors cut on all drones, mission halted')
        response.success = True
        response.message = 'emergency kill: motors cut on all drones'
        return response

    # ---- FSM -----------------------------------------------------
    def _tick(self):
        if self._killed:
            return

        now = time.monotonic()
        if self.state in {
            'TAKEOFF',
            'ALIGN_HOME',
            'START_COVERAGE',
            'COVERING',
            'RETURN_HOME',
        }:
            if self._check_flight_envelope(now):
                return

        if self.state == 'PREPARE':
            self._do_decompose()
            self._do_plan()
            self._publish_plan()
            self._begin_trajectory_uploads()
            self._set_state('UPLOADING_TRAJECTORIES')
        elif self.state == 'UPLOADING_TRAJECTORIES':
            if self._trajectory_uploads_finished():
                self._set_state('AWAITING_START')
        elif self.state == 'AWAITING_START':
            if self._start_requested:
                self._set_state('TAKEOFF')
        elif self.state == 'TAKEOFF':
            self._step_takeoff(now)
        elif self.state == 'ALIGN_HOME':
            self._step_align_home(now)
        elif self.state == 'START_COVERAGE':
            self._step_start_coverage(now)
        elif self.state == 'COVERING':
            self._step_covering(now)
        elif self.state == 'RETURN_HOME':
            self._step_return_home(now)
        elif self.state == 'LAND':
            self._step_land(now)
        elif self.state == 'SAFETY_LAND':
            self._step_safety_land(now)
        elif self.state == 'AWAITING_UGV_DONE':
            self.get_logger().info(
                f'aerial mission complete, '
                f'{len(self.detected_markers)} markers found: '
                f'{sorted(self.detected_markers.keys())} -- '
                'UGV completion wait not implemented; proceeding to DONE')
            self._set_state('DONE')
        elif self.state == 'DONE':
            pass

    def _do_decompose(self):
        # Single planner facade -- picks simple vs scopp from mission_map's
        # `planner` field (see mission_planner.plan_zones). Both launch files'
        # _compute_homes() call the exact same function so spawn points match
        # the plan this node flies. zone_cells stays the {col,row,x,y} dict
        # shape _publish_plan reads, whichever algorithm produced it.
        self._zone_plans = plan_zones(
            self.mission_map, self.drone_ids, self.dead_zone_margin)
        self.zone_cells = {d: p.cells for d, p in self._zone_plans.items()}

    @staticmethod
    def _compress_collinear_waypoints(waypoints):
        """Keep only start/end and direction-change points."""
        cleaned = []
        for point in waypoints:
            point = (float(point[0]), float(point[1]))
            if not cleaned or dist2d(cleaned[-1], point) > 1.0e-6:
                cleaned.append(point)

        if len(cleaned) <= 2:
            return cleaned

        compressed = [cleaned[0]]
        for index in range(1, len(cleaned) - 1):
            previous = cleaned[index - 1]
            current = cleaned[index]
            following = cleaned[index + 1]

            first_dx = current[0] - previous[0]
            first_dy = current[1] - previous[1]
            second_dx = following[0] - current[0]
            second_dy = following[1] - current[1]

            cross = first_dx * second_dy - first_dy * second_dx
            dot = first_dx * second_dx + first_dy * second_dy
            if abs(cross) > 1.0e-7 or dot <= 0.0:
                compressed.append(current)

        compressed.append(cleaned[-1])
        return compressed

    @staticmethod
    def _septic_coefficients(start_value, end_value, duration):
        """Degree-7 smoothstep: zero velocity/acceleration/jerk at both ends."""
        duration = max(float(duration), 1.0e-3)
        delta = float(end_value) - float(start_value)
        return [
            float(start_value),
            0.0,
            0.0,
            0.0,
            35.0 * delta / duration**4,
            -84.0 * delta / duration**5,
            70.0 * delta / duration**6,
            -20.0 * delta / duration**7,
        ]

    def _segment_duration(self, start_xy, end_xy):
        distance = dist2d(start_xy, end_xy)
        if distance <= 1.0e-6:
            return 0.0

        # max(ds/du) for 35u^4-84u^5+70u^6-20u^7 is 2.1875.
        # This keeps polynomial peak velocity near cruise_speed.
        speed_limited = 2.20 * distance / self.cruise_speed
        return max(float(self.min_leg_duration), speed_limited)

    def _build_trajectory(self, waypoints):
        flight_waypoints = self._compress_collinear_waypoints(
            waypoints)
        pieces = []
        total_duration = 0.0

        for start_xy, end_xy in zip(
            flight_waypoints[:-1],
            flight_waypoints[1:],
        ):
            duration = self._segment_duration(
                start_xy, end_xy)
            if duration <= 0.0:
                continue

            piece = TrajectoryPolynomialPiece()
            piece.duration = seconds_to_duration_msg(duration)
            piece.poly_x = self._septic_coefficients(
                start_xy[0], end_xy[0], duration)
            piece.poly_y = self._septic_coefficients(
                start_xy[1], end_xy[1], duration)
            piece.poly_z = [
                self.cruise_altitude,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            piece.poly_yaw = [0.0] * 8
            pieces.append(piece)
            total_duration += duration

        return flight_waypoints, pieces, total_duration

    def _do_plan(self):
        for drone_id, handle in self.drones.items():
            # Waypoints were already computed by the planner facade in
            # _do_decompose (simple: boustrophedon; scopp: NN/TSP path) --
            # waypoints[0] doubles as this drone's home, same contract for
            # both algorithms.
            handle.waypoints = self._zone_plans[drone_id].waypoints
            if handle.waypoints:
                home_xy = handle.waypoints[0]
                handle.home_position = (
                    float(home_xy[0]),
                    float(home_xy[1]),
                    0.0,
                )
                handle.last_target_xy = (
                    float(home_xy[0]),
                    float(home_xy[1]),
                )

            (
                handle.flight_waypoints,
                handle.trajectory_pieces,
                handle.trajectory_duration,
            ) = self._build_trajectory(handle.waypoints)

            handle.arrived_index = 0
            handle.done = len(handle.trajectory_pieces) == 0

            self.get_logger().info(
                f'{drone_id}: coverage cells={len(handle.waypoints)}, '
                f'flight turns={len(handle.flight_waypoints)}, '
                f'pieces={len(handle.trajectory_pieces)}, '
                f'duration={handle.trajectory_duration:.1f}s')

    def _begin_trajectory_uploads(self):
        for handle in self.drones.values():
            if handle.trajectory_pieces:
                handle.upload_trajectory()
            else:
                handle.upload_future = None
        self._upload_started = True
        self._upload_deadline = time.monotonic() + 30.0
        self.get_logger().info(
            'uploading onboard polynomial trajectories')

    def _trajectory_uploads_finished(self):
        if not self._upload_started:
            return False
        if time.monotonic() > self._upload_deadline:
            raise RuntimeError(
                'trajectory upload timed out after 30s')

        for handle in self.drones.values():
            future = handle.upload_future
            if future is None:
                continue
            if not future.done():
                return False
            exception = future.exception()
            if exception is not None:
                raise RuntimeError(
                    f'{handle.drone_id} trajectory upload failed: '
                    f'{exception}')

        self.get_logger().info(
            'all onboard trajectories uploaded')
        self._upload_started = False
        return True

    def _publish_plan(self):
        # Zone is visualized as one small square per assigned cell (rather than
        # a single merged polygon) -- matches the cell-based decomposition
        # itself and needs no polygon geometry to build.
        half = self.coverage_line_spacing / 2.0
        zone_array = ZoneAssignmentArray()
        for drone_id, cells in self.zone_cells.items():
            za = ZoneAssignment(drone_id=drone_id)
            for cell in cells:
                cx, cy = cell['x'], cell['y']
                pmsg = PolygonMsg()
                pmsg.points = [
                    Point32(x=float(cx - half), y=float(cy - half), z=0.0),
                    Point32(x=float(cx + half), y=float(cy - half), z=0.0),
                    Point32(x=float(cx + half), y=float(cy + half), z=0.0),
                    Point32(x=float(cx - half), y=float(cy + half), z=0.0),
                ]
                za.polygons.append(pmsg)
            zone_array.zones.append(za)
        self.zones_pub.publish(zone_array)

        path_array = CoveragePathArray()
        for drone_id, handle in self.drones.items():
            cp = CoveragePath(drone_id=drone_id)
            path_msg = Path()
            path_msg.header.frame_id = self.mission_map.get('frame_id', 'map')
            for x, y in handle.waypoints:
                pose = PoseStamped()
                pose.header.frame_id = path_msg.header.frame_id
                pose.pose.position.x = float(x)
                pose.pose.position.y = float(y)
                pose.pose.position.z = self.cruise_altitude
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            cp.path = path_msg
            path_array.paths.append(cp)
        self.paths_pub.publish(path_array)
        self.get_logger().info('published zone assignment and coverage paths')
        self._publish_progress()  # 0/total for every drone, before flight even starts

    def _step_takeoff(self, now):
        if not self._takeoff_sent:
            for handle in self.drones.values():
                handle.send_takeoff(
                    self.cruise_altitude,
                    self.takeoff_duration,
                )
            self._takeoff_sent = True
            self._phase_deadline = (
                now
                + self.takeoff_duration
                + self.takeoff_settle_time
            )
            self.get_logger().info(
                'onboard takeoff sent to all drones')
        elif now >= self._phase_deadline:
            self._align_sent = False
            self._set_state('ALIGN_HOME')

    def _step_align_home(self, now):
        """Align each drone with the first trajectory point using one slow go_to."""
        if not self._align_sent:
            max_duration = 0.0
            for handle in self.drones.values():
                home_xy = (
                    handle.home_position[0],
                    handle.home_position[1],
                )
                current_xy = self.live_xy.get(
                    handle.drone_id,
                    home_xy,
                )
                duration = self._segment_duration(
                    current_xy, home_xy)
                duration = max(
                    float(self.min_leg_duration),
                    duration,
                )
                handle.send_go_to(
                    (
                        home_xy[0],
                        home_xy[1],
                        self.cruise_altitude,
                    ),
                    0.0,
                    duration,
                )
                max_duration = max(
                    max_duration, duration)

            self._align_sent = True
            self._phase_deadline = (
                now
                + max_duration
                + self.leg_settle_margin
            )
            self.get_logger().info(
                'aligning drones to trajectory start points')
        elif now >= self._phase_deadline:
            self._set_state('START_COVERAGE')

    def _step_start_coverage(self, now):
        if not self._trajectory_start_sent:
            for handle in self.drones.values():
                handle.visited_mask = set()
                handle.arrived_index = 0
                handle.done = not bool(
                    handle.trajectory_pieces)
                handle.start_future = None
                if handle.trajectory_pieces:
                    handle.start_trajectory()

            self._trajectory_start_sent = True
            self._trajectory_start_request_time = now
            self.get_logger().info(
                'starting onboard coverage trajectories')
            return

        for handle in self.drones.values():
            future = handle.start_future
            if future is None:
                continue
            if not future.done():
                return
            exception = future.exception()
            if exception is not None:
                raise RuntimeError(
                    f'{handle.drone_id} trajectory start failed: '
                    f'{exception}')

        for handle in self.drones.values():
            handle.covering_start_time = (
                self._trajectory_start_request_time)

        self._trajectory_started = True
        self._publish_progress()
        self.get_logger().info(
            'all onboard coverage trajectories accepted')
        self._set_state('COVERING')

    def _publish_progress(self):
        array = DroneProgressArray()
        for handle in self.drones.values():
            array.progress.append(DroneProgress(
                drone_id=handle.drone_id,
                waypoint_index=handle.arrived_index,
                total_waypoints=len(handle.waypoints),
                visited_indices=sorted(handle.visited_mask),
            ))
        self.progress_pub.publish(array)

    def _update_visited_cells(self, handle):
        """Mark coverage cells only from the measured /states position.

        The onboard trajectory runs independently of GCS progress. Every
        unvisited cell is checked so one missed corner cannot block later
        cells from being reported correctly.
        """
        live_xy = self.live_xy.get(handle.drone_id)
        if live_xy is None:
            return False
        changed = False
        for idx, wp in enumerate(handle.waypoints):
            if idx in handle.visited_mask:
                continue
            if dist2d(live_xy, wp) <= self.arrival_radius:
                handle.visited_mask.add(idx)
                if idx > handle.arrived_index:
                    handle.arrived_index = idx
                changed = True
        return changed

    def _step_covering(self, now):
        """Monitor onboard trajectories; no periodic flight setpoints are sent."""
        progress_changed = False

        for handle in self.drones.values():
            if self._update_visited_cells(handle):
                progress_changed = True

            if handle.done:
                continue

            elapsed = now - handle.covering_start_time
            if elapsed >= (
                handle.trajectory_duration
                + self.trajectory_settle_time
            ):
                handle.done = True
                self.get_logger().info(
                    f'{handle.drone_id}: onboard trajectory complete '
                    f'after {elapsed:.1f}s')

        if progress_changed:
            self._publish_progress()

        if all(handle.done for handle in self.drones.values()):
            self._returning_sent = False
            self._set_state('RETURN_HOME')

    def _step_return_home(self, now):
        if not getattr(self, '_returning_sent', False):
            max_duration = 0.0
            for handle in self.drones.values():
                home_xy = (
                    handle.home_position[0],
                    handle.home_position[1],
                )
                current_xy = self.live_xy.get(
                    handle.drone_id,
                    handle.last_target_xy,
                )
                duration = max(
                    float(self.min_leg_duration),
                    self._segment_duration(
                        current_xy, home_xy),
                )
                handle.send_go_to(
                    (
                        home_xy[0],
                        home_xy[1],
                        self.cruise_altitude,
                    ),
                    0.0,
                    duration,
                )
                max_duration = max(
                    max_duration, duration)

            self._returning_sent = True
            self._phase_deadline = (
                now
                + max_duration
                + self.leg_settle_margin
            )
            self.get_logger().info(
                'returning to home positions with onboard go_to')
        elif now >= self._phase_deadline:
            self._land_sent = False
            self._set_state('LAND')

    def _step_land(self, now):
        if not self._land_sent:
            for handle in self.drones.values():
                handle.send_land(
                    0.02, self.land_duration)
            self._land_sent = True
            self._phase_deadline = (
                now
                + self.land_duration
                + self.land_settle_time
            )
            self.get_logger().info(
                'onboard land sent to all drones')
        elif now >= self._phase_deadline:
            self._set_state('AWAITING_UGV_DONE')

    def _flight_bounds(self):
        boundary = [
            tuple(point)
            for point in self.mission_map['boundary']
        ]
        xs = [point[0] for point in boundary]
        ys = [point[1] for point in boundary]
        margin = self.safety_boundary_margin
        return (
            min(xs) - margin,
            max(xs) + margin,
            min(ys) - margin,
            max(ys) + margin,
        )

    def _check_flight_envelope(self, now):
        """Repeated gross envelope violations trigger an onboard safety land."""
        min_x, max_x, min_y, max_y = self._flight_bounds()

        for drone_id, xyz in self.live_xyz.items():
            x, y, z = xyz
            stale = (
                now
                - self.live_state_time.get(drone_id, now)
            ) > 1.0
            if stale:
                continue

            violated = (
                x < min_x
                or x > max_x
                or y < min_y
                or y > max_y
                or z > self.safety_max_altitude
                or z < -0.20
            )

            if violated:
                self._safety_violations[drone_id] += 1
            else:
                self._safety_violations[drone_id] = 0

            if (
                self._safety_violations[drone_id]
                >= self.safety_violation_ticks
            ):
                self.get_logger().error(
                    f'{drone_id} left flight envelope at '
                    f'({x:.2f}, {y:.2f}, {z:.2f}); '
                    'commanding safety land')
                self._trigger_safety_land(now)
                return True

        return False

    def _trigger_safety_land(self, now):
        if self._safety_land_sent:
            return
        for handle in self.drones.values():
            handle.send_land(
                0.02, self.land_duration)
        self._safety_land_sent = True
        self._phase_deadline = (
            now
            + self.land_duration
            + self.land_settle_time
        )
        self._set_state('SAFETY_LAND')

    def _step_safety_land(self, now):
        if now >= self._phase_deadline:
            self.get_logger().error(
                'mission aborted after safety land')
            self._set_state('DONE')

    def _publish_markers(self):
        """Publishes the full current detected_markers set to /mission/markers
        (latched) -- called incrementally as each new marker is found (see
        _on_detection), so a UGV consumer sees markers as they're found
        instead of waiting for the whole aerial mission to finish."""
        array = MarkerRecordArray()
        for marker_id, (x, y, z) in self.detected_markers.items():
            array.markers.append(
                MarkerRecord(marker_id=marker_id, position=Point(x=x, y=y, z=z)))
        self.markers_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()