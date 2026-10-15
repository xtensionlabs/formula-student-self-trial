# fs_bringup

## Purpose
One launch file starts the whole stack; per-node parameters live in `config/`; RViz config in `rviz/`;
`viz_node` turns messages into RViz markers.

## Run
```bash
# Windows:  cd D:\FSDS ; .\FSDS.exe
# WSL terminal 1 (bridge):
source ~/fsds/ros2/install/setup.zsh
ros2 launch fsds_ros2_bridge fsds_ros2_bridge.launch.py
# WSL terminal 2 (our stack):
source /opt/ros/humble/setup.zsh; source ~/fsds/ros2/install/setup.zsh; source ~/fsds_ws/install/setup.zsh
ros2 launch fs_bringup autonomous.launch.py            # add rviz:=false to skip RViz
```
Launch details: `cone_detector_node` runs LiDAR-only until `models/cones.pt` exists; the ANN node loads
`models/ann_controller.pt`; the MPC node stays braked until the GO signal arrives (`/fsds/signal/go`).

## Config files (`config/`)
`state_estimator.yaml`, `perception.yaml`, `slam.yaml`, `loop_closure.yaml`, `planner.yaml`, `mpc.yaml`,
`ann.yaml`. Topic/frame names follow `D:\FSDS\settings.json` (`Lidar1`, `cam1`, vehicle `FSCar`); change
them there if you rename sensors. `perception.yaml` relaxes the cone height rules because `Lidar1` has only
4 lasers; `planner.yaml` uses conservative first-lap speeds (`v_max 8`, `a_lat_max 10`).

## RViz (`rviz/autonomous.rviz`, fixed frame `map`)
LiDAR pointcloud, perception cones and SLAM map as coloured markers (yellow/blue/orange/grey = unknown),
corrected map, centerline path, velocity arrows (length and colour = target speed), car pose with covariance.

## Parameters of viz_node
`arrow_every` (m between speed arrows), `v_max` (speed that maps to full green).

## Offline checks without FSDS
```bash
ros2 launch fs_bringup autonomous.launch.py rviz:=false &
python3 ~/fsds_ws/scripts/bag_replay.py synthetic --laps 2      # simulated cones + drifting odometry
python3 ~/fsds_ws/scripts/stack_check.py                        # rates and last values per topic
```
Headless result with the synthetic drive: 62/62 landmarks, loop closures from the second lap, centerline
188 poses at 4 Hz, `/slam/map_corrected` published, control commands published after GO.

## Known limitations
- The headless check is open loop (no plant), so MPC solves get slow/fail as the tracking error grows
  and the node correctly brakes / falls back; closed-loop behaviour is covered by `fs_control/sim.py` tests.
- Nothing here has run against the live simulator yet.
- `loop_closure_node` re-optimises the whole graph at every closing keyframe (~0.3 s at 60 keyframes).
