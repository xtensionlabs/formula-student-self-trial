# Architecture

Formula Student Driverless (FS-AI) stack for the FSDS simulator. ROS 2 Humble, Python 3.10.

```
 FSDS (Windows) --bridge--> /fsds/imu, /fsds/gss ------------------> fs_state_estimation (UKF)
                            /fsds/lidar/Lidar1 --+                        | /state/odometry (100 Hz)
                            /fsds/camera/cam1 ---+--> fs_perception        |
                            /tf_static ----------+    (LiDAR cones +      |
                                                       YOLO colour)       |
                                         /perception/cones |              |
                                                           v              v
                                                     fs_slam: slam_node (EKF-SLAM)
                              loop_closure_node (pose graph) |        | /slam/map, /slam/pose, TF map->FSCar
                                  /slam/map_corrected <------+        v
                                                     fs_planning: path_planner_node
                                                       | /planning/centerline, /planning/velocity_profile
                                                       v
                          fs_control: mpc_controller_node  <-- /control/ann_cmd -- ann_controller_node
                                                       | /fsds/control_command (20 Hz)
                                                       v
                                                    FSDS car
```

## Packages and where to read more
| Package | Nodes | Doc |
|---|---|---|
| fs_state_estimation | state_estimator_node | [STATE_ESTIMATION.md](STATE_ESTIMATION.md) |
| fs_perception | cone_detector_node (+ all custom msgs) | [PERCEPTION.md](PERCEPTION.md) |
| fs_slam | slam_node, loop_closure_node | [SLAM.md](SLAM.md) |
| fs_planning | path_planner_node | [PLANNING.md](PLANNING.md) |
| fs_control | mpc_controller_node, ann_controller_node | [CONTROL.md](CONTROL.md) |
| fs_bringup | viz_node, launch, config, RViz | [BRINGUP.md](BRINGUP.md) |

## Conventions
- **Frames**: `odom` (UKF, origin = start), `map` (SLAM, origin = start pose), `fsds/FSCar` (vehicle),
  `fsds/Lidar1`, `fsds/cam1` (sensors; static TF from the bridge). `slam_node` broadcasts `map -> fsds/FSCar`.
- **Messages**: Cone / ConeArray / SpeedProfile live in `fs_perception/msg`; Python helpers are in
  `fs_perception_py` (a module named `fs_perception` would collide with the generated message module).
- **Diagnostics**: every node reports its processing latency on `/diagnostics` (`LatencyDiag`).
- **Steering sign**: FSDS `+1` steers RIGHT; internally delta > 0 is LEFT; the conversion is only in
  `fs_control/ctrl_common.py` (`steer_sign`).
- **Parameters**: every tuning knob is a ROS parameter; values live in `fs_bringup/config/*.yaml`.
- **Logging**: `rclpy` logger only; scripts use `logging`.

## Design decisions (and why)
| Decision | Reason |
|---|---|
| NumPy/SciPy RANSAC + clustering instead of open3d | ~30 lines, no 400 MB dependency, easy to debug on sparse FSDS clouds |
| Own SE(2) pose graph (scipy) instead of g2o/GTSAM | tiny graph (keyframes), no build dependency |
| CasADi + IPOPT directly instead of do-mpc | same solver; single shared model/cost definition, warm start, hard CPU-time limit for the 45 ms budget |
| GP speed prior + forward/backward accel pass | GP gives uncertainty-aware safe speed; the pass makes the profile drivable (brake before corners) |
| GSS frame auto-detection | docs do not state the frame; a wrong guess would silently corrupt the estimate |
| Reentrant callback group for control inputs | slow MPC solves must not starve pose/path updates |

## Testing
`scripts/build.sh test` builds and runs `tests/` (27 tests): UKF, cone clustering/fusion, EKF-SLAM + loop
closure on a synthetic track, Delaunay centerline + speed profile, MPC cost / closed loop, ANN TorchScript,
and ROS integration of the controller nodes (GO gating, steering sign, ANN fallback).
`scripts/bag_replay.py` (record / play / synthetic) and `scripts/stack_check.py` allow offline checks.

## Status
- [x] Steps 1-8 implemented; headless end-to-end run on synthetic data verified (see BRINGUP.md).
- [x] ANN controller trained (`models/ann_controller.pt`); MPC weights tuned only in a smoke run.
- [ ] YOLO weights (`models/cones.pt`) not trained: the stack runs LiDAR-only with unknown colours.
- [ ] Not yet validated against the live FSDS simulator (frame names, GSS frame, LiDAR tuning, speed limits).

## Roadmap to be competitive
1. Validate in FSDS: sensor frames, cone detection range (denser LiDAR in `settings.json`), SLAM consistency.
2. Train YOLO on FSDS cones (ground truth from `/fsds/testing_only/track`), then enable colour fusion.
3. Replace the kinematic MPC model with a dynamic (tyre) model for the speed limit; retune with CMA-ES on
   recorded FSDS centerlines (`tune_mpc_cmaes.py --track`).
4. Autocross -> trackdrive logic: lap counting, switch to `/slam/map_corrected`, raise speed lap by lap.
5. Finish/EBS handling (`/fsds/signal/finished`).
