cmake_minimum_required(VERSION 3.8)
project(mission_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DroneState.msg"
  "msg/DroneProgress.msg"
  "msg/DroneProgressArray.msg"
  "msg/MarkerDetection.msg"
  "msg/ZoneAssignment.msg"
  "msg/ZoneAssignmentArray.msg"
  "msg/CoveragePath.msg"
  "msg/CoveragePathArray.msg"
  "msg/MarkerRecord.msg"
  "msg/MarkerRecordArray.msg"
  DEPENDENCIES std_msgs geometry_msgs nav_msgs
)

ament_package()
