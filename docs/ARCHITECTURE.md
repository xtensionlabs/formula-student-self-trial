# Architecture (living document)

Pipeline: sensors -> fs_state_estimation (UKF) + fs_perception (cones) -> fs_slam (map, pose)
-> fs_planning (centerline, speed) -> fs_control (MPC, ANN fallback) -> /cmd.

## Conventions
- All custom messages (Cone, ConeArray, SpeedProfile) live in `fs_perception/msg`.
  Python helpers/nodes of that package are in `fs_perception_py` (a module named
  `fs_perception` would collide with the generated message module).
- Every node publishes processing latency on `/diagnostics` via `fs_perception_py.diag.LatencyDiag`.
- Build: `colcon build --symlink-install` (source ROS Humble and `~/fsds/ros2/install` first).

## Status
- [x] Step 1-2: scaffold + messages (builds clean)
- [x] Step 3: fs_state_estimation (UKF, 6 unit tests, see STATE_ESTIMATION.md)
- [x] Step 4: fs_perception (LiDAR cones + YOLO + fusion; PERCEPTION.md). YOLO weights still to train.
- [x] Step 5: fs_slam (EKF-SLAM + pose-graph loop closure; SLAM.md; synthetic track tests)
- [x] Step 6: fs_planning (Delaunay centerline, GP speed profile + lap refinement; PLANNING.md)
- [x] Step 7: fs_control (CasADi MPC + TorchScript ANN fallback, CMA-ES tuner, training script; CONTROL.md)
