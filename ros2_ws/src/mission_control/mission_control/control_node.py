#!/usr/bin/env python3
"""Central mission state machine.

Single rclpy.Node that owns the whole mission from a cold start to landing:
loads the known mission map, splits it into 3 zones (one per drone), plans a
boustrophedon coverage path per zone, drives the 3 Crazyflies through
crazyswarm2, collects ArUco detections published by whichever cf_perception
node is running (sim or real, same topic contract), and publishes the final
marker list for the (future) UGV routing node to consume.

Coverage (the ㄹ-shape sweep) is flown as a single continuously-advancing
*streamed* position reference (crazyflie_interfaces/FullState on
/cfN/cmd_full_state) rather than a sequence of discrete go_to calls -- go_to
plans a smooth stop-to-stop trajectory for each leg (zero velocity at both
ends), so chaining many of them one per grid cell means literally
decelerating to a stop at every single cell before accelerating into the
next leg. Streaming a position that keeps sliding along the whole path at
cruise_speed (arc-length parameterized, see _interpolate_path) removes that
stop-and-go behavior entirely: takeoff/land/return-home still use the
high-level services (fire-and-forget, no action servers exist in
crazyswarm2), but coverage is a moving target the onboard/sim controller
tracks continuously.

Runs as a single non-blocking timer-driven FSM so that the /detections
subscriber keeps getting serviced while the drones are moving -- no blocking
sleeps anywhere in this node.
"""
import math
import time

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from builtin_interfaces.msg import Duration as DurationMsg
from crazyflie_interfaces.msg import FullState
from crazyflie_interfaces.srv import GoTo, Land, NotifySetpointsStop, Takeoff
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

from mission_control.mission_planner import plan_mission

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
    """Per-drone crazyswarm2 clients + coverage-following state."""

    def __init__(self, node, drone_id, home_position):
        self.node = node
        self.drone_id = drone_id
        self.home_position = home_position  # [x, y, z]
        prefix = '/' + drone_id
        self.takeoff_client = node.create_client(Takeoff, prefix + '/takeoff')
        self.land_client = node.create_client(Land, prefix + '/land')
        self.go_to_client = node.create_client(GoTo, prefix + '/go_to')
        self.notify_setpoints_stop_client = node.create_client(
            NotifySetpointsStop, prefix + '/notify_setpoints_stop')
        # Streaming position reference used during COVERING -- see module
        # docstring for why this replaces per-cell go_to calls.
        self.cmd_full_state_pub = node.create_publisher(FullState, prefix + '/cmd_full_state', 1)

        self.waypoints = []  # list of (x, y), z comes from cruise altitude separately
        self.cum_dist = [0.0]  # cum_dist[i] = arc length from waypoints[0] to waypoints[i]
        self.covering_start_time = None
        # visited_mask: set of waypoint indices whose cell the drone's *live
        # measured* position has actually come within arrival_radius of --
        # checked independently against every remaining waypoint each tick
        # (see ControlNode._update_visited_cells), not just the next one in
        # path order. That independence matters: if the drone cuts a corner
        # and never gets close enough to one cell, later cells still get
        # marked visited normally instead of being permanently blocked.
        self.visited_mask = set()
        # arrived_index: highwater mark (max visited index so far) -- used
        # for the "swept so far" tube/progress-bar display, where a
        # monotonic contiguous number is what's wanted even if visited_mask
        # itself has a gap.
        self.arrived_index = 0
        self.done = False
        self.last_target_xy = (home_position[0], home_position[1])

    def wait_for_services(self, node, timeout_sec):
        for client, name in (
            (self.takeoff_client, 'takeoff'),
            (self.land_client, 'land'),
            (self.go_to_client, 'go_to'),
            (self.notify_setpoints_stop_client, 'notify_setpoints_stop'),
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

    def send_cmd_full_state(self, xy, z):
        """Stream an absolute position reference (velocity/acceleration left at
        zero -- position error alone is enough for the onboard/sim controller
        to track a slowly, continuously moving target; see module docstring).
        """
        msg = FullState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(xy[0])
        msg.pose.position.y = float(xy[1])
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0  # identity quaternion -- no yaw control
        self.cmd_full_state_pub.publish(msg)
        self.last_target_xy = (xy[0], xy[1])

    def send_notify_setpoints_stop(self, remain_valid_millisecs=100):
        """Tell the firmware/sim streaming setpoints are ending, so it reverts
        to high-level command mode -- required before go_to()/land() will work
        again after cmd_full_state streaming (crazyswarm2 API contract)."""
        req = NotifySetpointsStop.Request()
        req.group_mask = 0
        req.remain_valid_millisecs = remain_valid_millisecs
        self.notify_setpoints_stop_client.call_async(req)


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

        self.drone_ids = list(self.get_parameter('drone_ids').value)
        self.min_leg_duration = self.get_parameter('min_leg_duration').value
        self.leg_settle_margin = self.get_parameter('leg_settle_margin').value
        self.takeoff_duration = self.get_parameter('takeoff_duration').value
        self.takeoff_settle_time = self.get_parameter('takeoff_settle_time').value
        self.land_duration = self.get_parameter('land_duration').value
        self.land_settle_time = self.get_parameter('land_settle_time').value
        self.arrival_radius = self.get_parameter('arrival_radius').value
        self.dead_zone_margin = self.get_parameter('dead_zone_margin').value

        mission_map_path = self.get_parameter('mission_map_path').value
        if not mission_map_path:
            raise RuntimeError(
                'mission_control requires the mission_map_path parameter '
                '(path to mission_map.yaml)')
        self.mission_map = self._load_mission_map(mission_map_path)
        self.cruise_altitude = float(self.mission_map['uav_cruise_altitude'])
        self.coverage_line_spacing = float(self.mission_map['coverage_line_spacing'])
        # Coverage streaming reference speed, m/s (control_node's
        # cmd_full_state advances along the path at this rate) -- kept in
        # mission_map.yaml, not a launch/ROS parameter, so it can just be
        # edited there like every other mission-tuning value (cruise
        # altitude, line spacing, ...). Lower it if cells are being missed:
        # the drone tracks a faster-moving reference more loosely, so it may
        # never come within arrival_radius of every cell center.
        self.cruise_speed = float(self.mission_map.get('cruise_speed', 0.3))
        # SCoPP path-planning profile: 'paper_nn' (greedy nearest-neighbour,
        # default) or 'metric_tsp' (grid shortest-path TSP, shorter routes but
        # a few seconds to plan on large zones). Edited in mission_map.yaml,
        # same as every other mission-tuning value. See path_planning.py.
        self.coverage_profile = self.mission_map.get('coverage_profile', 'paper_nn')

        self.drones = {}
        # home_position here is just a startup placeholder -- _do_plan() below
        # overwrites it with the start cell of whatever zone this drone ends up
        # allocated, once that's known (see mission_planner.plan_mission).
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
        # `or []` (not just `.get(..., [])`): a bare `dead_zones:` key with
        # nothing under it parses to None in YAML, which isn't caught by the
        # dict default and would crash the comprehension below.
        dead_zones = [
            [tuple(p) for p in dz['points']] for dz in (self.mission_map.get('dead_zones') or [])
        ]
        # Full SCoPP pipeline (grid -> Lloyd/auction allocation -> coverage
        # path) in one call, so control_node and the launch files compute the
        # exact same plan. _do_plan below just reads the stored result.
        self.mission_plans = plan_mission(
            boundary, dead_zones, self.coverage_line_spacing, self.drone_ids,
            dead_zone_margin=self.dead_zone_margin, profile=self.coverage_profile)
        self.zone_cells = {d: plan.cells for d, plan in self.mission_plans.items()}

    @staticmethod
    def _build_cum_dist(waypoints):
        """cum_dist[i] = arc length walked along the path from waypoints[0] to
        waypoints[i] -- lets _step_covering turn "elapsed time x cruise_speed"
        into a position along the whole ㄹ path with a simple lookup/interp."""
        cum = [0.0]
        for i in range(1, len(waypoints)):
            cum.append(cum[-1] + dist2d(waypoints[i - 1], waypoints[i]))
        return cum

    def _do_plan(self):
        # Home is *derived from* the assigned zone (its first cell), not the
        # other way around -- a drone assigned to the 3rd region should spawn
        # inside the 3rd region, not at some unrelated pre-given point that
        # then needs a long straight commute to reach its own zone.
        for drone_id, handle in self.drones.items():
            handle.waypoints = self.mission_plans[drone_id].waypoints
            if handle.waypoints:
                home_xy = handle.waypoints[0]
                handle.home_position = (home_xy[0], home_xy[1], 0.0)
                handle.last_target_xy = home_xy
            handle.cum_dist = self._build_cum_dist(handle.waypoints)
            handle.arrived_index = 0
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
                cx, cy = cell.center
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
        now = time.monotonic()
        for handle in self.drones.values():
            handle.covering_start_time = now
            handle.arrived_index = 0
            handle.visited_mask = set()
            handle.done = len(handle.waypoints) <= 1
            handle.last_target_xy = (handle.home_position[0], handle.home_position[1])
        self._publish_progress()

    @staticmethod
    def _interpolate_path(waypoints, cum_dist, target_dist):
        """Position `target_dist` meters along the polyline `waypoints`."""
        if target_dist <= 0.0:
            return waypoints[0]
        if target_dist >= cum_dist[-1]:
            return waypoints[-1]
        for i in range(1, len(cum_dist)):
            if cum_dist[i] >= target_dist:
                seg_len = cum_dist[i] - cum_dist[i - 1]
                t = 0.0 if seg_len <= 1e-9 else (target_dist - cum_dist[i - 1]) / seg_len
                x0, y0 = waypoints[i - 1]
                x1, y1 = waypoints[i]
                return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        return waypoints[-1]

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
        """Mark cells visited using the drone's *live measured* position
        (self.live_xy, from /states), not the commanded streaming reference.

        The reference in _step_covering slides continuously regardless of
        where the drone actually is, so deriving "visited" from the
        reference's own progress paints the GCS map ahead of the real
        drone -- exactly the "미리 채워지는" bug reported. Since control no
        longer waits for arrival to advance the reference (that coupling is
        what caused the earlier understeer/corner-cutting issue), gating
        *only the progress readout* on real position is now safe: it can't
        stall the reference, it just reports true visited cells.

        Every not-yet-visited waypoint is checked each tick (not just the
        next one in path order) so that cutting a corner and missing one
        cell doesn't permanently block every later cell from being marked
        visited -- only that one cell stays unvisited.
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
        """Stream each active drone's absolute position reference, sliding
        along its own ㄹ path at `cruise_speed` (arc-length parameterized) --
        see module docstring for why this replaces discrete per-cell go_to
        calls. Each drone advances completely independently of the others.
        Progress (for GCS "visited cell" painting) is derived separately,
        from live measured position -- see _update_visited_cells.
        """
        progress_changed = False
        for handle in self.drones.values():
            if handle.done:
                if self._update_visited_cells(handle):
                    progress_changed = True
                continue
            elapsed = now - handle.covering_start_time
            target_dist = elapsed * self.cruise_speed
            total_dist = handle.cum_dist[-1]
            if target_dist >= total_dist:
                target_dist = total_dist
                handle.done = True
                handle.send_notify_setpoints_stop()  # hand back to high-level mode

            ref_xy = self._interpolate_path(handle.waypoints, handle.cum_dist, target_dist)
            handle.send_cmd_full_state(ref_xy, self.cruise_altitude)

            if self._update_visited_cells(handle):
                progress_changed = True

        if progress_changed:
            self._publish_progress()

        if all(handle.done for handle in self.drones.values()):
            self._returning_sent = False
            # notify_setpoints_stop() needs a moment to actually take effect
            # before go_to() will be accepted again -- see _step_return_home.
            self._return_home_not_before = now + 0.2
            self._set_state('RETURN_HOME')

    def _step_return_home(self, now):
        # Give notify_setpoints_stop() (sent as each drone finished its
        # streamed coverage sweep) a brief moment to actually take effect --
        # go_to() is a high-level command and won't be honored until the
        # firmware/sim has reverted out of streaming-setpoint mode.
        if now < getattr(self, '_return_home_not_before', 0.0):
            return
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
