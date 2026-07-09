#!/usr/bin/env python3
"""Central mission state machine.

Single rclpy.Node that owns the whole mission from a cold start to landing:
loads the known mission map, splits it into 3 zones (one per drone), plans a
boustrophedon coverage path per zone, drives the 3 Crazyflies through
crazyswarm2's per-drone services directly (no action servers exist in
crazyswarm2 -- takeoff/land/go_to are fire-and-forget services), collects
ArUco detections published by whichever cf_perception node is running
(sim or real, same topic contract), and publishes the final marker list for
the (future) UGV routing node to consume.

Runs as a single non-blocking timer-driven FSM so that the /detections
subscriber keeps getting serviced while legs are "in flight" -- no blocking
sleeps anywhere in this node.
"""
import math
import time

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from builtin_interfaces.msg import Duration as DurationMsg
from crazyflie_interfaces.srv import GoTo, Land, Takeoff
from geometry_msgs.msg import Point, Point32
from geometry_msgs.msg import Polygon as PolygonMsg
from geometry_msgs.msg import PoseStamped
from mission_interfaces.msg import (
    CoveragePath, CoveragePathArray, DroneProgress, DroneProgressArray, DroneState,
    MarkerDetection, MarkerRecord, MarkerRecordArray, ZoneAssignment, ZoneAssignmentArray,
)
from nav_msgs.msg import Path
from std_msgs.msg import String
from std_srvs.srv import Trigger

from mission_control.coverage_plan import plan_coverage
from mission_control.zone_split import assign_cells_to_drones, build_cells

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
    """Per-drone crazyswarm2 service clients + coverage-following state."""

    def __init__(self, node, drone_id, home_position, home_yaw):
        self.drone_id = drone_id
        self.home_position = home_position  # [x, y, z]
        self.home_yaw = home_yaw
        prefix = '/' + drone_id
        self.takeoff_client = node.create_client(Takeoff, prefix + '/takeoff')
        self.land_client = node.create_client(Land, prefix + '/land')
        self.go_to_client = node.create_client(GoTo, prefix + '/go_to')

        self.waypoints = []  # list of (x, y), z comes from cruise altitude separately
        # arrived_index: last waypoint the drone has actually finished flying to
        # (index 0 == home, true immediately -- see plan_coverage). pending_index:
        # index it's currently *in flight toward*, becomes arrived_index only once
        # that leg's deadline passes. Published progress must reflect arrived_index,
        # not "index we just sent a command for" -- otherwise the GCS paints a
        # waypoint as visited the instant it's commanded, before the drone has
        # actually gotten anywhere near it.
        self.arrived_index = 0
        self.pending_index = 0
        self.done = False
        self.last_target_xy = (home_position[0], home_position[1])

    def wait_for_services(self, node, timeout_sec):
        for client, name in (
            (self.takeoff_client, 'takeoff'),
            (self.land_client, 'land'),
            (self.go_to_client, 'go_to'),
        ):
            if not client.wait_for_service(timeout_sec=timeout_sec):
                node.get_logger().warn(
                    f'{self.drone_id}/{name} service not available after {timeout_sec}s '
                    '(is the crazyflie_server launched?)')

    def send_takeoff(self, height, duration):
        req = Takeoff.Request()
        req.group_mask = 0
        req.height = float(height)
        req.duration = seconds_to_duration_msg(duration)
        self.takeoff_client.call_async(req)

    def send_land(self, height, duration):
        req = Land.Request()
        req.group_mask = 0
        req.height = float(height)
        req.duration = seconds_to_duration_msg(duration)
        self.land_client.call_async(req)

    def send_go_to(self, xyz, yaw_deg, duration):
        req = GoTo.Request()
        req.group_mask = 0
        req.relative = False
        req.goal = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
        req.yaw = float(yaw_deg)
        req.duration = seconds_to_duration_msg(duration)
        self.go_to_client.call_async(req)
        self.last_target_xy = (xyz[0], xyz[1])


class ControlNode(Node):

    def __init__(self):
        super().__init__('control_node')

        self.declare_parameter('mission_map_path', '')
        self.declare_parameter('cruise_speed', 0.3)
        self.declare_parameter('min_leg_duration', 1.5)
        self.declare_parameter('leg_settle_margin', 0.5)
        self.declare_parameter('takeoff_duration', 2.0)
        self.declare_parameter('takeoff_settle_time', 2.5)
        self.declare_parameter('land_duration', 2.5)
        self.declare_parameter('land_settle_time', 3.0)
        self.declare_parameter('start_immediately', False)
        self.declare_parameter('dead_zone_margin', 0.15)

        self.cruise_speed = self.get_parameter('cruise_speed').value
        self.min_leg_duration = self.get_parameter('min_leg_duration').value
        self.leg_settle_margin = self.get_parameter('leg_settle_margin').value
        self.takeoff_duration = self.get_parameter('takeoff_duration').value
        self.takeoff_settle_time = self.get_parameter('takeoff_settle_time').value
        self.land_duration = self.get_parameter('land_duration').value
        self.land_settle_time = self.get_parameter('land_settle_time').value
        self.dead_zone_margin = self.get_parameter('dead_zone_margin').value

        mission_map_path = self.get_parameter('mission_map_path').value
        if not mission_map_path:
            raise RuntimeError(
                'mission_control requires the mission_map_path parameter '
                '(path to mission_map.yaml)')
        self.mission_map = self._load_mission_map(mission_map_path)
        self.cruise_altitude = float(self.mission_map['uav_cruise_altitude'])
        self.coverage_line_spacing = float(self.mission_map['coverage_line_spacing'])

        self.drones = {}
        for d in self.mission_map['drones']:
            handle = DroneHandle(self, d['id'], d['home_position'], d.get('home_yaw', 0.0))
            handle.wait_for_services(self, timeout_sec=5.0)
            self.drones[d['id']] = handle

        self.zones_pub = self.create_publisher(
            ZoneAssignmentArray, '/mission/zones', LATCHED_QOS)
        self.paths_pub = self.create_publisher(
            CoveragePathArray, '/mission/coverage_paths', LATCHED_QOS)
        self.markers_pub = self.create_publisher(
            MarkerRecordArray, '/mission/markers', LATCHED_QOS)
        self.state_pub = self.create_publisher(String, '/mission/state', 10)
        # Authoritative "how far along its path is each drone" feed for the GCS --
        # this used to be *guessed* client-side from nearest-waypoint distance,
        # which is unreliable on a zig-zag path (many waypoints can be spatially
        # close to the current position without actually being "next"). control_node
        # already tracks the real wp_index itself, so it's simplest to just publish it.
        self.progress_pub = self.create_publisher(DroneProgressArray, '/mission/progress', 10)

        self.detections_sub = self.create_subscription(
            MarkerDetection, '/detections', self._on_detection, 10)
        self.detected_markers = {}  # marker_id -> (x, y, z)

        # Real measured position per drone (from /states, same feed cf_perception
        # publishes for the GCS). Used instead of blindly assuming a drone has
        # already reached the last commanded target -- if sim/real timing runs
        # faster or slower than our wall-clock leg-duration estimate, computing
        # the next leg's distance from a *wrong* assumed start point produces a
        # too-short duration for the real distance still left to cover, which
        # crazyswarm2's own go_to docs warn drives an unstable/runaway trajectory.
        self.states_sub = self.create_subscription(
            DroneState, '/states', self._on_drone_state, 20)
        self.live_xy = {}  # drone_id -> (x, y)

        self._start_requested = self.get_parameter('start_immediately').value
        self.start_srv = self.create_service(Trigger, '/mission/start', self._on_start_request)

        self.state = 'DECOMPOSE'
        self._phase_deadline = None
        self._leg_deadline = None
        self._takeoff_sent = False
        self._land_sent = False

        self.timer = self.create_timer(0.1, self._tick)
        self.get_logger().info('control_node started, state=DECOMPOSE')

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
    def _on_drone_state(self, msg):
        self.live_xy[msg.drone_id] = (msg.position.x, msg.position.y)

    def _on_detection(self, msg):
        if msg.marker_id not in self.detected_markers:
            self.get_logger().info(
                f'marker {msg.marker_id} detected by {msg.drone_id} at '
                f'({msg.position.x:.2f}, {msg.position.y:.2f})')
        self.detected_markers[msg.marker_id] = (
            msg.position.x, msg.position.y, msg.position.z)

    def _on_start_request(self, request, response):
        self._start_requested = True
        response.success = True
        response.message = 'mission start requested'
        return response

    # ---- FSM -----------------------------------------------------
    def _tick(self):
        now = time.monotonic()
        if self.state == 'DECOMPOSE':
            self._do_decompose()
            self._set_state('PLAN')
        elif self.state == 'PLAN':
            self._do_plan()
            self._set_state('PUBLISH_PLAN')
        elif self.state == 'PUBLISH_PLAN':
            self._publish_plan()
            self._set_state('AWAITING_START')
        elif self.state == 'AWAITING_START':
            if self._start_requested:
                self._set_state('TAKEOFF')
        elif self.state == 'TAKEOFF':
            self._step_takeoff(now)
        elif self.state == 'COVERING':
            self._step_covering(now)
        elif self.state == 'RETURN_HOME':
            self._step_return_home(now)
        elif self.state == 'LAND':
            self._step_land(now)
        elif self.state == 'PUBLISH_MARKERS':
            self._publish_markers()
            self._set_state('DONE')
        elif self.state == 'DONE':
            pass

    def _do_decompose(self):
        boundary = [tuple(p) for p in self.mission_map['boundary']]
        dead_zones = [
            [tuple(p) for p in dz['points']] for dz in self.mission_map.get('dead_zones', [])
        ]
        cells = build_cells(
            boundary, dead_zones, self.coverage_line_spacing, self.dead_zone_margin)
        self.zone_cells = assign_cells_to_drones(cells, self.mission_map['drones'])

    def _do_plan(self):
        for drone_id, handle in self.drones.items():
            cells = self.zone_cells[drone_id]
            start_xy = (handle.home_position[0], handle.home_position[1])
            handle.waypoints = plan_coverage(cells, start_xy)
            handle.arrived_index = 0
            handle.pending_index = 0
            handle.done = len(handle.waypoints) <= 1  # index 0 is home itself, nothing to fly

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
                handle.send_takeoff(self.cruise_altitude, self.takeoff_duration)
            self._takeoff_sent = True
            self._phase_deadline = now + self.takeoff_duration + self.takeoff_settle_time
            self.get_logger().info('takeoff sent to all drones')
        elif now >= self._phase_deadline:
            self._init_covering()
            self._set_state('COVERING')

    def _init_covering(self):
        for handle in self.drones.values():
            handle.arrived_index = 0
            handle.pending_index = 0
            handle.done = len(handle.waypoints) <= 1
            handle.last_target_xy = (handle.home_position[0], handle.home_position[1])
        self._send_next_leg_for_all(time.monotonic())

    def _send_next_leg_for_all(self, now):
        """Advance every still-active drone by one waypoint.

        Only called once the *previous* leg's deadline has passed (from
        _step_covering, or once at covering start from _init_covering), so
        `pending_index` -- the target we sent last time -- can now be trusted
        as actually reached: promote it to `arrived_index` before picking the
        next target. Publishing progress using `arrived_index` (rather than
        "whichever index we just commanded") is what makes the GCS's visited-
        path fill in as the drone actually gets there, instead of the instant
        the command is sent.
        """
        max_duration = 0.0
        any_active = False
        for handle in self.drones.values():
            if handle.done:
                continue
            handle.arrived_index = handle.pending_index
            next_idx = handle.arrived_index + 1
            if next_idx >= len(handle.waypoints):
                handle.done = True
                continue
            target_xy = handle.waypoints[next_idx]
            current_xy = self.live_xy.get(handle.drone_id, handle.last_target_xy)
            leg_dist = dist2d(current_xy, target_xy)
            duration = max(self.min_leg_duration, leg_dist / self.cruise_speed)
            handle.send_go_to(
                (target_xy[0], target_xy[1], self.cruise_altitude), 0.0, duration)
            handle.last_target_xy = target_xy
            handle.pending_index = next_idx
            max_duration = max(max_duration, duration)
            any_active = True

        self._publish_progress()

        if not any_active:
            self._returning_sent = False
            self._set_state('RETURN_HOME')
            return
        self._leg_deadline = now + max_duration + self.leg_settle_margin

    def _publish_progress(self):
        array = DroneProgressArray()
        for handle in self.drones.values():
            array.progress.append(DroneProgress(
                drone_id=handle.drone_id,
                waypoint_index=handle.arrived_index,
                total_waypoints=len(handle.waypoints),
            ))
        self.progress_pub.publish(array)

    def _step_covering(self, now):
        if now >= self._leg_deadline:
            self._send_next_leg_for_all(now)

    def _step_return_home(self, now):
        if not getattr(self, '_returning_sent', False):
            max_duration = 0.0
            for handle in self.drones.values():
                home_xy = (handle.home_position[0], handle.home_position[1])
                current_xy = self.live_xy.get(handle.drone_id, handle.last_target_xy)
                leg_dist = dist2d(current_xy, home_xy)
                duration = max(self.min_leg_duration, leg_dist / self.cruise_speed)
                handle.send_go_to(
                    (home_xy[0], home_xy[1], self.cruise_altitude), 0.0, duration)
                handle.last_target_xy = home_xy
                max_duration = max(max_duration, duration)
            self._returning_sent = True
            self._phase_deadline = now + max_duration + self.leg_settle_margin
            self.get_logger().info('returning to home positions')
        elif now >= self._phase_deadline:
            self._land_sent = False
            self._set_state('LAND')

    def _step_land(self, now):
        if not self._land_sent:
            for handle in self.drones.values():
                handle.send_land(0.02, self.land_duration)
            self._land_sent = True
            self._phase_deadline = now + self.land_duration + self.land_settle_time
            self.get_logger().info('land sent to all drones')
        elif now >= self._phase_deadline:
            self._set_state('PUBLISH_MARKERS')

    def _publish_markers(self):
        array = MarkerRecordArray()
        for marker_id, (x, y, z) in self.detected_markers.items():
            array.markers.append(
                MarkerRecord(marker_id=marker_id, position=Point(x=x, y=y, z=z)))
        self.markers_pub.publish(array)
        self.get_logger().info(
            f'mission complete, {len(array.markers)} markers found: '
            f'{sorted(self.detected_markers.keys())}')


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
