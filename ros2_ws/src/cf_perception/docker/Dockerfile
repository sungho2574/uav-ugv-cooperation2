# ============================================================
# cf_perception real_perception_node -- Jetson container
#
# Base: ultralytics/ultralytics:latest-jetson-jetpack6 (Ubuntu 22.04 + CUDA +
# a JetPack-matched torch/ultralytics build), + ROS2 Humble on top.
#
# Why this exists: a JetPack-matched torch/ultralytics build is fragile and
# slow to install natively on a Jetson (NVIDIA's own wheel index, tied to the
# exact JetPack version), whereas this base image already has it working.
# real_perception_node is the only node in this project that needs YOLO/
# torch -- control_node, gcs_dashboard, and crazyswarm2 itself stay native on
# the Jetson host and talk to this container purely over ROS2 DDS
# (network_mode: host, see docker-compose.yml / real.launch.py's
# perception_runtime: "docker" path).
#
# Build context MUST be the OUTER ros2_ws/src/ that this whole repo is
# checked out into alongside crazyswarm2 -- i.e. the layout docs/RUNNING.md
# already requires for native builds too:
#
#   ros2_ws/src/
#     crazyswarm2/               <- provides crazyflie_interfaces/
#     uav-ugv-cooperation2/      <- this repo (has its OWN nested ros2_ws/src/)
#       ros2_ws/src/cf_perception/mission_interfaces/...
#
# crazyflie_interfaces isn't part of this repo, so it can only be reached by
# pointing the build context at that outer src/ (see docker-compose.yml's
# `context: ../../../../..`, five levels up from this file, and
# `dockerfile:` pointing back in) -- COPY can't reach outside its context no
# matter where inside this repo the Dockerfile itself lives.
#
#   cd ros2_ws/src/cf_perception/docker
#   docker compose build
# ============================================================
FROM ultralytics/ultralytics:latest-jetson-jetpack6

# ---- ROS2 Humble apt repository ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg lsb-release \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) \
        signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-rclpy \
        ros-humble-std-msgs \
        ros-humble-std-srvs \
        ros-humble-builtin-interfaces \
        ros-humble-geometry-msgs \
        ros-humble-nav-msgs \
        ros-humble-sensor-msgs \
        ros-humble-rosidl-default-generators \
        ros-humble-rosidl-default-runtime \
        ros-humble-cv-bridge \
        python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# torch/ultralytics/opencv-python/numpy are expected to already be present
# from the base image's own pip install (JetPack-matched CUDA wheels) --
# fail loudly here, at build time, rather than letting the colcon build step
# below silently let setuptools pull generic (non-Jetson, CPU-only)
# replacement wheels from PyPI if it thinks any of these are missing.
RUN python3 -c "import cv2, numpy, torch, ultralytics; print('cv2', cv2.__version__, 'torch', torch.__version__)"
# PyYAML is the one cf_perception dependency the base image doesn't already
# carry -- pure Python, no CUDA/arch concerns, safe to fetch generically.
RUN pip3 install --no-cache-dir PyYAML

# ---- Build message + perception packages ----
# Build context is the OUTER ros2_ws/src/ (see header) -- crazyswarm2/ and
# uav-ugv-cooperation2/ are both direct children of it.
WORKDIR /ros_ws/src
COPY uav-ugv-cooperation2/ros2_ws/src/mission_interfaces/ mission_interfaces/
COPY crazyswarm2/crazyflie_interfaces/ crazyflie_interfaces/
COPY uav-ugv-cooperation2/ros2_ws/src/cf_perception/ cf_perception/

WORKDIR /ros_ws
RUN bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --packages-select mission_interfaces crazyflie_interfaces cf_perception \
        --cmake-args -DCMAKE_BUILD_TYPE=Release"

# PARAMS_FILE is provided at `docker run` time (see real.launch.py's
# perception_runtime: "docker" path, which generates a fresh per-mission
# params yaml and bind-mounts it in at this same path so the container can
# read it directly) -- /params.yaml is just a manual-testing fallback.
CMD ["bash", "-c", \
    "source /opt/ros/humble/setup.bash && \
     source /ros_ws/install/setup.bash && \
     ros2 run cf_perception real_perception_node --ros-args --params-file ${PARAMS_FILE:-/params.yaml}"]
