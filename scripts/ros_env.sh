#!/usr/bin/env bash
# Source this in every terminal that runs ROS for the FSDS stack (bridge included):
#   source ~/fsds_ws/scripts/ros_env.sh
# Keep the default Fast DDS transports: multicast does not work on this WSL setup, local discovery relies on
# shared memory. After killing ROS processes, clear stale segments:  rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_*
source /opt/ros/humble/setup.bash
source ~/fsds/ros2/install/setup.bash          # fs_msgs + FSDS bridge
source ~/fsds_ws/install/setup.bash            # this workspace
