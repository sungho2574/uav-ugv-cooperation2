import os
import shutil
import subprocess
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessIO
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from mission_control.coverage_plan import plan_coverage
from mission_control.zone_split import assign_cells_to_drones, build_cells

# Must match control_node's `dead_zone_margin` parameter default -- see
# sim.launch.py's _compute_homes for why both sides compute independently.
DEAD_ZONE_MARGIN = 0.15

# Default YOLO weights, resolved via cf_perception's installed share/weights/
# dir (see cf_perception/setup.py's data_files) rather than a source-tree
# path, so it resolves correctly regardless of the launching cwd. Used
# whenever mission_map.yaml's `yolo.weights_path` is left empty; set that
# field to override with a different model without touching this file.
DEFAULT_YOLO_WEIGHTS_FILENAME = 'human_yolo11n_gray.onnx'

# The cflib backend creates ROS services before both Crazyflies have
# necessarily finished TOC/log/parameter/memory initialization. Starting
# trajectory upload during that interval can saturate the shared radio and
# produce `Too many packets lost`. Start mission_control only after the
# crazyflie_server reports that every Crazyflie is fully connected.
CRAZYFLIES_READY_TEXT = b'All Crazyflies are fully connected!'
PROCESS_IO_BUFFER_BYTES = 2048


def _enabled_drone_ids(crazyflies_cfg):
    """Same source as sim.launch.py -- see there for why."""
    return [
        drone_id for drone_id, robot_cfg in crazyflies_cfg.get('robots', {}).items()
        if robot_cfg.get('enabled', True)
    ]


def _compute_homes(mission_map, drone_ids):
    """Same computation as sim.launch.py -- see there for why."""
    boundary = [tuple(p) for p in mission_map['boundary']]
    # See sim.launch.py for why `or []` (not just `.get(..., [])`) is needed.
    dead_zones = [[tuple(p) for p in dz['points']] for dz in (mission_map.get('dead_zones') or [])]
    cells = build_cells(
        boundary, dead_zones, mission_map['coverage_line_spacing'], DEAD_ZONE_MARGIN)
    zone_cells = assign_cells_to_drones(cells, drone_ids)
    homes = {}
    for drone_id in drone_ids:
        waypoints = plan_coverage(zone_cells[drone_id])
        homes[drone_id] = waypoints[0] if waypoints else (0.0, 0.0)
    return homes


def _generate_crazyflies_yaml(crazyflies_cfg, homes, base_crazyflies_path):
    """Same `initial_position` injection as sim.launch.py, PLUS (real-only,
    hence not shared with sim.launch.py) seeding each drone's onboard kalman
    position estimate to match wherever it's actually been placed.

    Unlike the sim backend, crazyswarm2's real (cflib) backend never reads
    `initial_position` at all -- it's declared but unused there, so every
    real Crazyflie's kalman filter boots assuming it's at world (0,0,0)
    regardless of physical placement. Without mocap (mocap:=False here),
    there's no external position source to correct that, so a mission that
    physically staged 3 drones at their 3 different zone-start cells would
    otherwise have all 3 report/believe they're at the same origin.

    The fix is the standard no-mocap Crazyflie technique: push
    kalman.initialX/Y/Z (the physical spot the drone was placed at) then
    kalman.resetEstimation=1 as per-robot firmware_params -- crazyswarm2
    already applies `robots.<id>.firmware_params` on connect (see
    crazyflie_server.py's _init_parameters, which takes per-robot values
    over robot_types/all), in TOC-sorted param-name order within a group,
    which happens to sort resetEstimation after the three initial* values --
    so this doesn't need any new code upstream, just this config.
    """
    for drone_id, (x, y) in homes.items():
        if drone_id in crazyflies_cfg.get('robots', {}):
            robot_cfg = crazyflies_cfg['robots'][drone_id]
            robot_cfg['initial_position'] = [x, y, 0.0]
            firmware_params = robot_cfg.setdefault('firmware_params', {})
            kalman_params = firmware_params.setdefault('kalman', {})
            kalman_params['initialX'] = x
            kalman_params['initialY'] = y
            kalman_params['initialZ'] = 0.0
            kalman_params['resetEstimation'] = 1

    fd, generated_path = tempfile.mkstemp(prefix='crazyflies_generated_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump(crazyflies_cfg, f)
    return generated_path


def _build_docker_perception_process(mission_map, perception_share, perception_params):
    """perception_runtime: "docker" path -- runs real_perception_node inside
    the cf_perception:jetson container (see cf_perception/docker/) instead of
    as a native ROS2 node, for hosts (the Jetson) where a JetPack-matched
    torch/ultralytics build is fragile to install directly. See
    mission_map.yaml's perception_runtime comment for the native/docker
    tradeoff and cf_perception/docker/Dockerfile for image details.

    Unlike docker-compose.yml (a manual-testing convenience only),
    this builds a plain `docker run` invocation directly so it can bind-mount
    a fresh, mission-specific params.yaml (generated here from the exact same
    perception_params dict the native path would hand to Node()) on every
    launch, plus the camera intrinsics / YOLO weights files at their own
    real, unmodified host paths -- host and container share the same
    filesystem on the Jetson, so no path translation is needed, just mount
    each file at an identical path on both sides.

    Does NOT build the image itself -- a fresh build is far too slow to
    happen inline on every mission launch. Build it once ahead of time with
    `cd cf_perception/docker && docker compose build`
    and this raises a clear error if that was never done.
    """
    docker_bin = shutil.which('docker')
    if docker_bin is None:
        raise RuntimeError(
            "perception_runtime is 'docker' but no `docker` binary was found on PATH")

    docker_image = mission_map.get('docker_image', 'cf_perception:jetson')
    inspect = subprocess.run(
        [docker_bin, 'image', 'inspect', docker_image],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if inspect.returncode != 0:
        raise RuntimeError(
            f"perception_runtime is 'docker' but image '{docker_image}' was not found -- "
            'build it first with `cd ros2_ws/src/cf_perception/docker && docker compose build`')

    fastdds_xml_path = os.path.join(perception_share, 'docker', 'fastdds_udp.xml')

    # ROS2 params yaml, `/**:` wildcard so it applies regardless of the
    # node's actual name/namespace -- same values the native Node(...) path
    # would receive, just serialized instead of passed in-process.
    fd, params_path = tempfile.mkstemp(prefix='real_perception_params_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump({'/**': {'ros__parameters': perception_params}}, f)

    cmd = [
        docker_bin, 'run', '--rm',
        '--network=host', '--ipc=host', '--runtime=nvidia',
        '-e', f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')}",
        '-e', f"RMW_IMPLEMENTATION={os.environ.get('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')}",
        '-e', 'FASTRTPS_DEFAULT_PROFILES_FILE=/fastdds_udp.xml',
        '-e', f'PARAMS_FILE={params_path}',
        '-v', f'{fastdds_xml_path}:/fastdds_udp.xml:ro',
        '-v', f'{params_path}:{params_path}:ro',
    ]
    # Bind-mount at the same absolute path on both sides -- only if the file
    # actually exists on this host (e.g. a custom yolo.weights_path set in
    # mission_map.yaml that hasn't been copied to this particular machine
    # yet would otherwise make `docker run` itself fail on a bad -v spec).
    for path in (perception_params['camera_intrinsics_path'], perception_params['yolo_weights_path']):
        if os.path.exists(path):
            cmd += ['-v', f'{path}:{path}:ro']
    cmd.append(docker_image)

    return ExecuteProcess(cmd=cmd, output='screen')


def _build(context, *args, **kwargs):
    bringup_share = get_package_share_directory('mission_bringup')
    perception_share = get_package_share_directory('cf_perception')
    mission_map_path = os.path.join(bringup_share, 'config', 'mission_map.yaml')
    base_crazyflies_path = os.path.join(bringup_share, 'config', 'crazyflies.yaml')
    ai_deck_ips_path = os.path.join(bringup_share, 'config', 'ai_deck_ips.yaml')
    camera_intrinsics_path = os.path.join(
        perception_share, 'config', 'camera_intrinsics.yaml')

    with open(base_crazyflies_path, 'r') as f:
        crazyflies_cfg = yaml.safe_load(f)
    drone_ids = _enabled_drone_ids(crazyflies_cfg)

    with open(mission_map_path, 'r') as f:
        mission_map = yaml.safe_load(f)
    homes = _compute_homes(mission_map, drone_ids)
    generated_crazyflies_path = _generate_crazyflies_yaml(
        crazyflies_cfg, homes, base_crazyflies_path)

    with open(ai_deck_ips_path, 'r') as f:
        ai_deck_ips = yaml.safe_load(f)
    missing = [d for d in drone_ids if d not in ai_deck_ips]
    if missing:
        raise RuntimeError(
            f'ai_deck_ips.yaml is missing an entry for: {missing} -- add each '
            "drone's AI-deck WiFi AP IP there before launching on real hardware")
    wifi_ips = [ai_deck_ips[d] for d in drone_ids]

    crazyswarm2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('crazyflie'), 'launch', 'launch.py')),
        launch_arguments={
            'backend': 'cflib',
            'crazyflies_yaml_file': generated_crazyflies_path,
            'mocap': 'False',
            'gui': 'False',
            'teleop': 'False',
            'rviz': 'False',
        }.items(),
    )

    control_node = Node(
        package='mission_control',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=[{
            'mission_map_path': mission_map_path,
            'drone_ids': drone_ids,
            'start_immediately': False,
        }],
    )

    yolo_cfg = mission_map.get('yolo') or {}
    default_yolo_weights_path = os.path.join(
        perception_share, 'weights', DEFAULT_YOLO_WEIGHTS_FILENAME)
    yolo_weights_path = yolo_cfg.get('weights_path') or default_yolo_weights_path
    perception_params = {
        'drone_ids': drone_ids,
        'wifi_ips': wifi_ips,
        'wifi_port': 5000,
        'marker_size': mission_map.get(
            'marker_size', 0.14),
        'camera_intrinsics_path': (
            camera_intrinsics_path),
        'camera_pitch_degs': [
            float(value)
            for value in yolo_cfg.get(
                'camera_pitch_degs', [45.0])
        ],
        'camera_latency_sec': float(
            yolo_cfg.get(
                'camera_latency_sec', 0.0)),
        'pose_tolerance_sec': float(
            yolo_cfg.get(
                'pose_tolerance_sec', 0.30)),
        'detection_backend': mission_map.get(
            'detection_backend', 'aruco'),
        'yolo_weights_path': yolo_weights_path,

        # Neural detector and official per-drone BoT-SORT.
        'yolo_confidence_threshold': float(
            yolo_cfg.get(
                'confidence_threshold', 0.35)),
        'yolo_low_confidence_threshold': float(
            yolo_cfg.get(
                'low_confidence_threshold', 0.10)),
        'yolo_nms_threshold': float(
            yolo_cfg.get(
                'nms_threshold', 0.45)),
        'yolo_image_size': int(
            yolo_cfg.get('image_size', 416)),
        'yolo_maximum_detections': int(
            yolo_cfg.get(
                'maximum_detections', 20)),
        'yolo_force_grayscale': bool(
            yolo_cfg.get(
                'force_grayscale', True)),
        'yolo_use_clahe': bool(
            yolo_cfg.get('use_clahe', False)),
        'yolo_track_buffer': int(
            yolo_cfg.get('track_buffer', 240)),
        'yolo_match_threshold': float(
            yolo_cfg.get(
                'match_threshold', 0.80)),
        'yolo_new_track_threshold': float(
            yolo_cfg.get(
                'new_track_threshold', 0.40)),

        # Projection.
        'yolo_target_height': float(
            yolo_cfg.get('target_height', 0.0)),
        'yolo_max_ground_range': float(
            yolo_cfg.get(
                'max_ground_range', 3.0)),
        'yolo_min_downward_ray': float(
            yolo_cfg.get(
                'min_downward_ray', 0.08)),

        # Static-object identity registry.
        'registry_sample_window': int(
            yolo_cfg.get(
                'registry_sample_window', 40)),
        'registry_confirmation_hits': int(
            yolo_cfg.get(
                'registry_confirmation_hits', 10)),
        'registry_minimum_confirmation_age': float(
            yolo_cfg.get(
                'registry_minimum_confirmation_age',
                1.0,
            )),
        'registry_maximum_ray_residual': float(
            yolo_cfg.get(
                'registry_maximum_ray_residual',
                0.22,
            )),
        'registry_ray_inlier_threshold': float(
            yolo_cfg.get(
                'registry_ray_inlier_threshold',
                0.24,
            )),
        'registry_ray_match_threshold': float(
            yolo_cfg.get(
                'registry_ray_match_threshold',
                0.30,
            )),
        'registry_minimum_parallax_deg': float(
            yolo_cfg.get(
                'registry_minimum_parallax_deg',
                5.0,
            )),
        'registry_minimum_baseline': float(
            yolo_cfg.get(
                'registry_minimum_baseline',
                0.30,
            )),
        'registry_minimum_triangulation_inliers': int(
            yolo_cfg.get(
                'registry_minimum_triangulation_inliers',
                7,
            )),
        'registry_covariance_floor': float(
            yolo_cfg.get(
                'registry_covariance_floor',
                0.20,
            )),
        'registry_association_chi2_gate': float(
            yolo_cfg.get(
                'registry_association_chi2_gate',
                16.0,
            )),
        'registry_association_max_distance': float(
            yolo_cfg.get(
                'registry_association_max_distance',
                0.60,
            )),
        'registry_ambiguity_chi2_gate': float(
            yolo_cfg.get(
                'registry_ambiguity_chi2_gate',
                36.0,
            )),
        'registry_ambiguity_max_distance': float(
            yolo_cfg.get(
                'registry_ambiguity_max_distance',
                1.20,
            )),
        'registry_duplicate_minimum_bbox_iou': float(
            yolo_cfg.get(
                'registry_duplicate_minimum_bbox_iou',
                0.15,
            )),
        'registry_duplicate_maximum_pixel_distance': float(
            yolo_cfg.get(
                'registry_duplicate_maximum_pixel_distance',
                24.0,
            )),
        'registry_distinct_evidence_frames': int(
            yolo_cfg.get(
                'registry_distinct_evidence_frames',
                5,
            )),
        'registry_distinct_maximum_bbox_iou': float(
            yolo_cfg.get(
                'registry_distinct_maximum_bbox_iou',
                0.05,
            )),
        'registry_distinct_minimum_pixel_distance': float(
            yolo_cfg.get(
                'registry_distinct_minimum_pixel_distance',
                35.0,
            )),
        'registry_bundle_inlier_threshold': float(
            yolo_cfg.get(
                'registry_bundle_inlier_threshold',
                0.30,
            )),
        'registry_bundle_minimum_group_inlier_ratio': float(
            yolo_cfg.get(
                'registry_bundle_minimum_group_inlier_ratio',
                0.55,
            )),
        'registry_bundle_maximum_group_median_error': float(
            yolo_cfg.get(
                'registry_bundle_maximum_group_median_error',
                0.30,
            )),
        'registry_appearance_merge_threshold': float(
            yolo_cfg.get(
                'registry_appearance_merge_threshold',
                0.12,
            )),
        'registry_appearance_max_distance': float(
            yolo_cfg.get(
                'registry_appearance_max_distance',
                1.20,
            )),
        'registry_hypothesis_spatial_gate': float(
            yolo_cfg.get(
                'registry_hypothesis_spatial_gate',
                0.45,
            )),
        'registry_hypothesis_minimum_tracklets': int(
            yolo_cfg.get(
                'registry_hypothesis_minimum_tracklets',
                3,
            )),
        'registry_hypothesis_minimum_separation': float(
            yolo_cfg.get(
                'registry_hypothesis_minimum_separation',
                0.55,
            )),
        'registry_hypothesis_separation_chi2': float(
            yolo_cfg.get(
                'registry_hypothesis_separation_chi2',
                25.0,
            )),
        'registry_hypothesis_timeout': float(
            yolo_cfg.get(
                'registry_hypothesis_timeout',
                45.0,
            )),
        'registry_tracklet_timeout': float(
            yolo_cfg.get(
                'registry_tracklet_timeout', 30.0)),
    }

    perception_runtime = mission_map.get('perception_runtime', 'native')
    if perception_runtime == 'native':
        perception_node = Node(
            package='cf_perception',
            executable='real_perception_node',
            name='real_perception_node',
            output='screen',
            parameters=[perception_params],
        )
    elif perception_runtime == 'docker':
        perception_node = _build_docker_perception_process(
            mission_map, perception_share, perception_params)
    else:
        raise RuntimeError(
            f"mission_map.yaml's perception_runtime is '{perception_runtime}' -- "
            "must be 'native' or 'docker'")

    gcs_node = Node(
        package='gcs_dashboard',
        executable='gcs_node',
        name='gcs_node',
        output='screen',
        parameters=[{
            'drone_ids': drone_ids,
            'mission_map_path': mission_map_path,
            'port': 5000,
        }],
    )

    # ProcessIO text may arrive in arbitrary chunks, so keep a short rolling
    # buffer and search across chunk boundaries. The handler is intentionally
    # global because crazyflie_server is created inside an included launch
    # description and is therefore not directly available here as a target
    # action. The message is unique to crazyflie_server.
    process_io_buffer = bytearray()
    control_started = {'value': False}

    def _start_control_when_ready(event):
        if control_started['value']:
            return None

        process_io_buffer.extend(event.text)
        if len(process_io_buffer) > PROCESS_IO_BUFFER_BYTES:
            del process_io_buffer[:-PROCESS_IO_BUFFER_BYTES]

        if CRAZYFLIES_READY_TEXT not in process_io_buffer:
            return None

        control_started['value'] = True
        print(
            '[mission_bringup] All Crazyflies fully connected; '
            'starting control_node now.',
            flush=True,
        )
        return [control_node]

    start_control_on_ready = RegisterEventHandler(
        OnProcessIO(
            on_stdout=_start_control_when_ready,
            on_stderr=_start_control_when_ready,
        )
    )

    return [
        start_control_on_ready,
        crazyswarm2_launch,
        perception_node,
        gcs_node,
    ]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=_build)])