# FSDS autonomous stack (Formula Student Driverless)

ROS 2 Humble workspace: perception, UKF state estimation, EKF-SLAM with loop closure, Delaunay planning
with a GP speed profile, MPC control with an ANN fallback. Start with [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```bash
scripts/build.sh test                 # build (colcon --symlink-install) + 27 unit/integration tests
ros2 launch fs_bringup autonomous.launch.py      # whole stack (see docs/BRINGUP.md)
python3 scripts/train_ann_controller.py          # retrain the ANN fallback from MPC rollouts
python3 scripts/tune_mpc_cmaes.py                # CMA-ES tuning of the MPC weights
python3 scripts/bag_replay.py synthetic          # feed the stack without FSDS
```

Layout: `src/` packages, `models/` weights, `scripts/` offline tools, `tests/`, `docs/` (one file per module).
