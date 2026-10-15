#!/usr/bin/env bash
# Build the workspace and (optionally) run the unit tests:  scripts/build.sh [test]
set -e
cd "$(dirname "$0")/.."
source /opt/ros/humble/setup.bash
source ~/fsds/ros2/install/setup.bash          # fs_msgs / FSDS bridge
colcon build --symlink-install 2>&1 | grep -E "Summary|failed|rror|Failed" || true
source install/setup.bash
[ "$1" = "test" ] && python3 -m pytest -q tests
true
