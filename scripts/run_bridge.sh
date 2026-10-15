#!/usr/bin/env bash
# Run the FSDS ROS 2 bridge and restart it whenever it dies (it aborts on a simulator RPC timeout).
#   scripts/run_bridge.sh
source ~/fsds_ws/scripts/ros_env.sh
while true; do
  ros2 launch fsds_ros2_bridge fsds_ros2_bridge.launch.py UDP_control:=true
  echo "[run_bridge] bridge exited, restarting in 2 s"
  sleep 2
done
