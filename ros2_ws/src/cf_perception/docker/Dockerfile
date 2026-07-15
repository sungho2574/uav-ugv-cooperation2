# cf_perception real_perception_node -- Jetson container
FROM ultralytics/ultralytics:8.3.51-jetson-jetpack6
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
# Dependencies must be installed at image-build time.  This prevents
# Ultralytics from running pip inside the live mission container.
ENV YOLO_AUTOINSTALL=false

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

RUN python3 -c "import cv2, numpy, torch, ultralytics; print('cv2', cv2.__version__, 'torch', torch.__version__)"
# `lap` is required by Ultralytics' Hungarian assignment.  Pin it here so the
# first real flight never pauses for an AutoUpdate/restart cycle.
RUN pip3 install --no-cache-dir PyYAML "lap==0.5.13"

WORKDIR /ros_ws/src
COPY uav-ugv-cooperation2/ros2_ws/src/mission_interfaces/ mission_interfaces/
COPY crazyswarm2/crazyflie_interfaces/ crazyflie_interfaces/
COPY uav-ugv-cooperation2/ros2_ws/src/cf_perception/ cf_perception/

WORKDIR /ros_ws
RUN pip3 install --no-cache-dir "setuptools==58.2.0"
RUN bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --packages-select mission_interfaces crazyflie_interfaces cf_perception \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros2cli \
        ros-humble-ros2run \
    && rm -rf /var/lib/apt/lists/*

CMD ["bash", "-c", \
    "source /opt/ros/humble/setup.bash && \
     source /ros_ws/install/setup.bash && \
     ros2 run cf_perception real_perception_node --ros-args --params-file ${PARAMS_FILE:-/params.yaml}"]