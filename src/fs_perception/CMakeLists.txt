cmake_minimum_required(VERSION 3.8)
project(fs_perception)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# All custom messages of the stack live here (one msg package, fewer deps).
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Cone.msg" "msg/ConeArray.msg" "msg/SpeedProfile.msg"
  DEPENDENCIES std_msgs geometry_msgs)

# Python code (nodes + shared helpers) installed next to the generated msgs.
ament_python_install_package(fs_perception_py)
install(PROGRAMS scripts/cone_detector_node DESTINATION lib/${PROJECT_NAME})
ament_package()
