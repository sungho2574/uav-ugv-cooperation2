# 실행 가이드

`mission_bringup` 기준 빌드/실행 방법과, 실행 전에 반드시 확인해야 할 사항을 정리한다. 패키지/노드/토픽 구조 자체는 [README.md](../README.md)를 참고.

## 목차

1. [사전 준비물](#1-사전-준비물)
2. [빌드](#2-빌드)
3. [실행 전 체크리스트 (공통)](#3-실행-전-체크리스트-공통)
4. [시뮬레이션 실행](#4-시뮬레이션-실행)
5. [실기체 실행 전 추가 체크리스트](#5-실기체-실행-전-추가-체크리스트)
   - 5-1. [젯슨에서 실행할 때 (Docker로 YOLO 백엔드 돌리기)](#5-1-젯슨에서-실행할-때-docker로-yolo-백엔드-돌리기)
6. [실기체 실행](#6-실기체-실행)
7. [대시보드 사용법](#7-대시보드-사용법)
8. [상태 확인용 CLI 명령](#8-상태-확인용-cli-명령)
9. [트러블슈팅](#9-트러블슈팅)

---

## 1. 사전 준비물

이 저장소는 **macOS 개발 머신에서 작성만 되었고 빌드/실행은 검증되지 않았다** — 아래는 Jetson 등 실제 ROS 2 Linux 환경에서 진행해야 한다.

- ROS 2 (Humble 이상 권장)
- [crazyswarm2](https://github.com/IMRCLab/crazyswarm2) 및 그 의존 패키지(`crazyflie`, `crazyflie_py`, `crazyflie_sim`, `crazyflie_interfaces`)가 같은 워크스페이스 안에 소스로 존재하고 빌드 가능해야 함
- `cv_bridge` (ROS 2 vision_opencv, apt로 설치: `ros-<distro>-cv-bridge`)
- Python 패키지: `shapely`, `PyYAML`, `numpy`, `opencv-python`(또는 `opencv-contrib-python`, `cv2.aruco` 필요), `flask`, `ultralytics`(YOLO 백엔드용 -- 모델을 학습/export한 것과 같은 라이브러리로 직접 추론함. `cv2.dnn`은 YOLO11의 attention 블록을 import하지 못해 쓰지 않음, [map_configuration.md](map_configuration.md) 8.4 참고. `ultralytics`는 `torch` 등 무거운 의존성을 함께 설치하니 용량/설치 시간 여유를 두고 설치할 것)
- 실기체 사용 시: AI-deck가 붙은 Crazyflie 3대, 같은 WiFi 네트워크, Crazyradio PA 동글

```bash
pip install shapely PyYAML numpy opencv-python flask ultralytics
```

## 2. 빌드

이 저장소의 `ros2_ws`를 crazyswarm2 소스가 있는 워크스페이스와 **같은 `src/` 아래에** 두거나, crazyswarm2를 별도로 빌드해 오버레이한다.

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

`mission_interfaces`는 `rosidl_generate_interfaces`를 쓰므로 처음 빌드 시 다른 패키지보다 먼저 메시지가 생성된다. 빌드 에러가 `crazyflie_interfaces`/`crazyflie_py`를 못 찾는다면 crazyswarm2가 같은 워크스페이스에 없다는 뜻이니 `--packages-up-to`나 오버레이 구성을 확인한다.

**`cf_perception`/`gcs_dashboard`(ament_python) 빌드가 `error: option --editable not recognized`로 죽는 경우**: 코드 문제가 아니라 `setuptools`가 최신 버전으로 올라가면서 `colcon build --symlink-install`이 쓰는 legacy `setup.py develop --editable` 경로와 안 맞아 생기는 잘 알려진 ROS2(Humble/Jammy) 호환 문제다. `torch`/`ultralytics`처럼 무거운 패키지를 pip로 새로 설치하면 `setuptools`가 딸려 올라오면서 재발하기 쉽다.

```bash
pip install setuptools==58.2.0 --user
colcon build --symlink-install
```

## 3. 실행 전 체크리스트 (공통)

`ros2_ws/src/mission_bringup/config/mission_map.yaml` 확인:

- [ ] `boundary`가 CCW(반시계) 순서인지
- [ ] `dead_zones`가 `boundary` 내부에 완전히 포함되는지 (경계를 벗어나면 zone 분할이 깨질 수 있음)
- [ ] `drones` 3대의 `home_position`이 서로 다른 x좌표를 가지는지 — `zone_split.py`는 홈 위치의 x좌표 순서로 zone을 배정하므로, x좌표가 같으면 배정이 임의 순서가 될 수 있음 (예제 지도처럼 y로만 구분해도 동작은 하지만, x로도 분리해두면 더 직관적)
- [ ] `uav_cruise_altitude`가 실제 비행 공간 천장(모션캡처 볼륨, 실내 층고 등)보다 충분히 낮은지
- [ ] `coverage_line_spacing`이 드론 카메라 시야각/속도에 비해 너무 좁거나 넓지 않은지 (너무 좁으면 waypoint 수가 급증해 임무 시간이 길어짐)

`ros2_ws/src/mission_bringup/config/crazyflies.yaml` 확인:

- [ ] `robots` 아래 각 드론의 `uri`(라디오 채널/주소)가 실제 장비와 일치하는지
- [ ] `initial_position`은 런치 시점에 `mission_map.yaml`의 `home_position`으로 **자동 덮어쓰기** 되므로 이 파일에 직접 적힌 값은 신경 쓸 필요 없음(README 9절 참고) — 실기체라면 그래도 물리적으로 드론을 그 `home_position` 위치에 놓아야 함
- [ ] `enabled: true`인 드론만 실제로 연결을 시도하니, 3대 모두 `true`인지

## 4. 시뮬레이션 실행

시뮬레이션 전용 추가 확인:

- [ ] `ros2_ws/src/mission_bringup/config/true_markers.yaml`의 마커 좌표가 전부 `boundary` 내부, `dead_zones` 바깥에 있는지 (경계 밖/장애물 안에 있으면 드론이 절대 도달 못 해 영원히 미검출로 남음)
- [ ] 마커 좌표가 각 드론의 zone과 최소 `grid_resolution`(기본 0.1 m) 이상 떨어져 있지 않은 경우, 스윕 라인이 지나가는 경로상에 실제로 있는지 (zone에는 속하지만 boustrophedon 스캔라인 사이 간격에 끼어 있으면 놓칠 수 있음 — `coverage_line_spacing`을 좁히거나 좌표를 스캔라인에 가깝게 조정)

실행:

```bash
ros2 launch mission_bringup sim.launch.py
```

정상 기동 시 콘솔에 대략 다음 순서로 로그가 뜬다: crazyflie_server(sim) 기동 → `control_node started, state=PREPARE` → `mission state: PREPARE -> AWAITING_START` → (대시보드에서 Start 클릭 후) `TAKEOFF` → `COVERING` → 마커 검출 로그(발견 즉시 `/mission/markers` 갱신) → `RETURN_HOME` → `LAND` → `AWAITING_UGV_DONE` → `aerial mission complete, N markers found` → `DONE`.

## 5. 실기체 실행 전 추가 체크리스트

- [ ] `real.launch.py`의 `wifi_ips` 플레이스홀더(`192.168.4.1/2/3`)를 각 AI-deck 실제 WiFi IP로 교체
- [ ] `ros2_ws/src/cf_perception/config/camera_intrinsics.yaml`을 실제 카메라로 체커보드 캘리브레이션한 값으로 교체 (기본값은 미보정 placeholder — 그대로 쓰면 마커/객체 world 좌표가 부정확함)
- [ ] `cf_perception/cf_perception/real_perception_node.py`의 `R_CAM_TO_BODY` (45도 하향 장착 가정)가 실제 AI-deck 장착각과 일치하는지 확인, 다르면 회전행렬 재계산
- [ ] `mission_map.yaml`의 `detection_backend`가 원하는 값(`aruco`/`yolo`)인지 확인. `yolo`인 경우 `yolo.weights_path`가 실제 존재하는 `.onnx` 파일을 가리키는지, `yolo.target_height`/`yolo.cluster_radius`가 실제 물체 배치와 맞는지 확인 (자세한 튜닝은 [map_configuration.md](map_configuration.md) 8절)
- [ ] `mission_map.yaml`의 `perception_runtime`이 이 머신에 맞는 값인지 확인 (`native` 기본값 / 젯슨은 `docker` — 아래 5-1절 참고). `docker`인 경우 이미지를 미리 빌드해뒀는지 확인
- [ ] 각 Crazyflie 배터리 충전 상태, 프로펠러 상태 확인
- [ ] `crazyflies.yaml`의 `firmware_params.commander.enHighLevel: 1`이 설정되어 있는지 (고수준 명령 `go_to`/`takeoff`/`land` 사용에 필수)
- [ ] 모션캡처 없이 온보드 상태추정만 사용하므로(`mocap:=False`), 장시간 비행 시 위치 드리프트가 누적될 수 있음 — 임무 영역 크기와 예상 비행 시간을 고려해 사전에 지상에서 위치 정확도를 확인
- [ ] Crazyradio PA가 3대 모두와 통신 가능한 범위/채널인지 확인
- [ ] 비상 정지 방법(조종기, `ros2 service call /all/emergency std_srvs/srv/Empty` 등) 숙지 후 이륙

### 5-1. 젯슨에서 실행할 때 (Docker로 YOLO 백엔드 돌리기)

젯슨은 JetPack 버전에 맞는 torch/ultralytics 빌드를 네이티브로 설치하기 까다로워서, `real_perception_node`만 따로 컨테이너(`cf_perception:jetson`, base: `ultralytics/ultralytics:latest-jetson-jetpack6`)에서 돌릴 수 있게 해뒀다. `control_node`/`gcs_dashboard`/crazyswarm2는 그대로 젯슨 호스트에서 네이티브로 돌고, 컨테이너는 `network_mode: host`로 붙어서 같은 ROS2 DDS로 통신한다 (`cf_perception/docker/fastdds_udp.xml`로 SHM 대신 UDP 강제).

1. 아래처럼 이 저장소(`uav-ugv-cooperation2`)와 `crazyswarm2`가 **같은 이름으로, 같은 상위 `ros2_ws/src/` 밑에** 나란히 존재해야 한다 (1절 사전 준비물과 동일 요구사항). 도커 빌드 컨텍스트가 이 바깥쪽 `ros2_ws/src/`라서 `crazyflie_interfaces`를 거기서 같이 COPY함 — 폴더 이름이 다르면 `docker compose build`가 `crazyflie_interfaces`를 못 찾고 실패한다.
   ```
   ros2_ws/src/
     crazyswarm2/               <- crazyflie_interfaces 제공
     uav-ugv-cooperation2/      <- 이 저장소 (자체 ros2_ws/src/를 안에 갖고 있음)
   ```
2. 이미지를 한 번 빌드해둔다 (미션 launch 시점에 자동 빌드 안 됨 — 매번 하기엔 너무 느림):
   ```bash
   cd ros2_ws/src/cf_perception/docker
   docker compose build
   ```
3. `mission_map.yaml`의 `perception_runtime: "docker"`로 설정 (기본은 `"native"`).
4. 평소처럼 `ros2 launch mission_bringup real.launch.py` 실행 — `real_perception_node`만 자동으로 `docker run`으로 뜨고, 나머지는 네이티브로 뜬다 (`real.launch.py`의 `_build_docker_perception_process` 참고). 실행 시점에 그때그때 `drone_ids`/`wifi_ips`/`detection_backend`/`yolo.*` 값을 담은 params yaml을 생성해서 컨테이너에 마운트하므로, `mission_map.yaml`을 고치면 다음 launch부터 바로 반영된다.

이미지를 안 빌드해뒀거나 `docker` 바이너리가 없으면 launch 시점에 명확한 에러 메시지로 실패한다 (조용히 네이티브로 폴백하지 않음).

## 6. 실기체 실행

```bash
ros2 launch mission_bringup real.launch.py
```

## 7. 대시보드 사용법

1. 브라우저로 `http://<gcs_node가 떠 있는 호스트>:5000` 접속 (Jetson에서 직접 열람 시 `http://localhost:5000`)
2. 좌측 3D 뷰에서 경계(초록 선), dead-zone(빨강 반투명), 3개 zone(드론별 색) 및 계획된 커버리지 경로(점선)가 보이는지 확인
3. 우측 영상 3개 확인 (시뮬은 "no signal"이 정상, 실기체는 실시간 영상이 떠야 함)
4. 상단 **Start Mission** 버튼 클릭 → `AWAITING_START → TAKEOFF` 전이 확인
5. 비행 중 드론 아이콘 이동, 마커 검출 시 노란 구+ID 라벨 표시, 방문한 경로 구간이 굵은 실선으로 바뀌는지 확인
6. 착륙 완료 후 상단 phase 텍스트가 `DONE`으로 바뀌는지 확인

## 8. 상태 확인용 CLI 명령

```bash
# 현재 미션 phase
ros2 topic echo /mission/state

# 계획된 zone/경로가 정상 발행됐는지 (latched라 나중에 붙어도 즉시 수신됨)
ros2 topic echo /mission/zones --once
ros2 topic echo /mission/coverage_paths --once

# 실시간 드론 위치 / 마커 검출
ros2 topic echo /states
ros2 topic echo /detections

# 임무 종료 후 최종 마커 목록
ros2 topic echo /mission/markers --once

# 대시보드 없이 CLI로 미션 시작
ros2 service call /mission/start std_srvs/srv/Trigger
```

## 9. 트러블슈팅

| 증상 | 확인할 것 |
|---|---|
| `control_node`가 시작하자마자 예외로 죽음 | `mission_map_path` 파라미터가 올바른 절대경로로 전달됐는지 (launch 파일에서 `get_package_share_directory` 사용, 빌드 후 재실행했는지) |
| 로그에 `takeoff service not available` 경고 | crazyswarm2의 `crazyflie_server`가 아직 기동 전이거나 실패함 — `ros2 node list`로 `crazyflie_server` 존재 확인, `crazyflies.yaml`의 `uri`/`enabled` 재확인 |
| 시뮬에서 `/states`, `/detections`가 전혀 안 나옴, `/cfN/pose`는 `topic list`엔 있지만 `echo`/`hz`엔 아무것도 안 나옴 | sim 백엔드는 `/cfN/pose`를 발행하지 않는 게 정상 (`crazyflie_sim`의 알려진 동작 — `sim_perception_node`가 `/tf`를 tf2로 조회하도록 되어 있음). `ros2 topic echo /tf`에 `world`→`cf1` 등 변환이 오는지, `crazyflies.yaml`의 `reference_frame`이 `sim_perception_node`의 `world_frame` 파라미터(기본 `world`)와 일치하는지 확인 |
| 시뮬에서 마커가 하나도 안 잡힘 | `true_markers.yaml` 좌표가 `boundary`/`dead_zones`와 실제로 겹치지 않는지, `coverage_line_spacing`이 마커를 지나치는 간격은 아닌지 |
| GCS 3D 뷰에 zone/경로가 안 보임 | `/mission/zones`, `/mission/coverage_paths`가 실제 발행됐는지(`ros2 topic echo ... --once`), gcs_node가 해당 토픽에 대해 같은 QoS(Transient Local)로 구독 중인지(코드 그대로면 문제 없음) |
| 실기체 영상은 뜨는데 마커 world 좌표가 명백히 틀림 | `camera_intrinsics.yaml` 캘리브레이션 여부, `R_CAM_TO_BODY` 장착각 가정, `/cfN/pose`와 프레임 캡처 시각 차이(0.2초 동기화 허용오차) 확인 |
| `detection_backend: yolo`로 실행했는데 `RuntimeError: yolo_weights_path is empty` 등으로 죽음 | `mission_map.yaml`의 `yolo.weights_path`가 비어있거나 잘못된 경로 — 실제 존재하는 `.onnx` 파일의 절대경로로 설정 |
| YOLO 백엔드에서 물체가 하나도 안 잡히거나 좌표가 이상함 | [map_configuration.md](map_configuration.md) 8절(특히 8.4 onnx export 형식 가정) 참고 |
| `perception_runtime: docker`인데 launch가 `image ... was not found`로 죽음 | 5-1절대로 `cd ros2_ws/src/cf_perception/docker && docker compose build`를 먼저 실행했는지 확인 (자동 빌드 안 함) |
| `perception_runtime: docker`인데 `no \`docker\` binary was found on PATH`로 죽음 | 젯슨에 Docker Engine + NVIDIA Container Toolkit 설치 여부 확인 |
| 도커 컨테이너의 `real_perception_node`가 떠도 `control_node`/`gcs_dashboard`에서 `/states`, `/detections`, `/mission/link_status`가 안 보임 | 컨테이너가 호스트와 같은 `ROS_DOMAIN_ID`/`RMW_IMPLEMENTATION`을 쓰는지(launch가 호스트 환경변수를 그대로 전달함) 확인, `docker run` 로그에 FastDDS 관련 에러가 있는지 확인 (`fastdds_udp.xml`이 `/fastdds_udp.xml`로 잘 마운트됐는지) |
| 드론 위치가 시간이 지날수록 dashboard에서 서서히 어긋남 | 모션캡처 없이 온보드 추정치만 쓰는 구조의 알려진 한계(드리프트) — 임무 범위/시간을 줄이거나 외부 포지셔닝 시스템 도입 검토 |
| 이륙 후 커버리지 도중 드론이 갑자기 비정상적으로 빠르게 튀어서 지도 밖으로 나가 멈춤 | `control_node`가 `/states`(실측 위치)를 받고 있는지 확인(`ros2 topic hz /states`) — 못 받으면 구간 소요시간을 "마지막 명령 목표에 이미 도착했다"는 가정만으로 계산해서, 실제 위치가 뒤처져 있을 때 `go_to`에 남은 거리에 비해 너무 짧은 `duration`이 전달되어 crazyswarm2가 불안정한 궤적을 계획할 수 있음. 그래도 재현되면 `cruise_speed`를 낮추거나 `min_leg_duration`/`leg_settle_margin`을 늘려서 여유를 더 주고 재시도 |
| GCS를 한동안 켜두면 브라우저 탭이 죽음(백엔드는 계속 살아있음) | 오래된 버전의 알려진 버그(수정 완료) — zone/경로를 매 폴링(300ms)마다 THREE.js 지오메트리를 새로 만들면서 이전 것을 `dispose()`하지 않아 GPU 메모리가 계속 쌓이던 문제였음. 최신 `app.js`는 데이터가 실제로 바뀔 때만 다시 그리고 나머지는 교체 전 `dispose()`를 호출함. 이 증상이 재현되면 `git pull` 등으로 최신 `static/app.js`가 반영됐는지부터 확인 |
