# fs_control

## Purpose
- **`mpc_controller_node`** – nonlinear MPC (`mpc.py`): kinematic bicycle, state `[x, y, psi, v]`,
  input `[delta, a]`, horizon N=20 at dt=0.05 s, `delta` in +-0.436 rad (FSDS limit, 25 deg; spec asked 0.5),
  `a` in [-10, 5] m/s^2. Cost per stage: cross-track error^2, heading error `2(1-cos)`, speed error^2,
  input magnitude, input *change* (smoothness). CasADi + IPOPT are used directly instead of the do-mpc
  wrapper (same solver underneath): one definition of model and cost shared with tests/tuner, warm start,
  hard CPU-time limit equal to the 45 ms budget.
- **`ann_controller_node`** – TorchScript MLP `[cte, heading_err, speed_err, curvature] -> [delta, a]`
  (`scripts/train_ann_controller.py`, imitates the MPC, loaded from `models/ann_controller.pt`).
- **Fallback (never silent):** if the MPC solve fails or exceeds `solve_limit_ms` (45), that step uses the
  newest ANN command (`/control/ann_cmd`, max age `ann_max_age`) and a WARN with a running counter is
  logged. With no fresh ANN command it logs an ERROR and brakes.
- **Pose**: `/slam/pose` is slow, `/state/odometry` fast. `ControlInputs` stores the map<-odom correction at
  every SLAM pose and returns `T (+) odom_now`, a smooth high-rate pose in the map frame. The MPC also
  predicts the state one step ahead to compensate computation delay.
- **Reference** (`reference.py`): nearest path point, then reference points spaced by the current speed
  along the path, with target speed from the planner (open paths taper to `v_end` at the end).
- **Command mapping**: FSDS steering `+1` = RIGHT (our delta > 0 is LEFT) so `steer_sign = -1`;
  `steering = sign*delta/d_max`, throttle = a/a_max, brake = a/a_min.

## Subscribed / published topics
| Topic | Type | Direction |
|---|---|---|
| `/slam/pose`, `/state/odometry` | PoseWithCovarianceStamped, Odometry | in |
| `/planning/centerline`, `/planning/velocity_profile` | Path, SpeedProfile | in |
| `/fs_msgs/msg/GoSignal` (param `go_topic`; FSDS bridge: `/fsds/signal/go`) | fs_msgs/GoSignal | in (MPC) |
| `/control/ann_cmd` | fs_msgs/ControlCommand | ANN out, MPC in |
| `/cmd` (param `cmd_topic`; FSDS bridge: `/fsds/control_command`) | fs_msgs/ControlCommand | MPC out, 20 Hz |
| `/diagnostics` | latency | out |

## Parameters
Common: topics, `d_max`, `steer_sign`, `a_max`, `a_min`. MPC: `rate_hz`, `horizon`, `dt`, `solve_limit_ms`,
`ann_max_age`, `require_go`, `wheelbase`, `lr`, and the cost weights `w.q_ct, w.q_head, w.q_v, w.r_d, w.r_a,
w.r_dd, w.r_da`. ANN: `model_path`, `ann_topic`, `rate_hz`.

## Offline tools
- `scripts/tune_mpc_cmaes.py` – CMA-ES over the 7 weights (log-space), parallel closed-loop evaluation in
  `fs_control/sim.py` (kinematic plant with actuator lag, wheelbase/accel mismatch, pose noise). Use
  `--track recorded.npz` (`xy`, `v`) to tune on a centerline recorded from FSDS. Output yaml goes to
  `fs_bringup/config/mpc.yaml` (copy the `w.*` values).
- `scripts/train_ann_controller.py` – collects MPC data (noisy execution, clean labels), trains the MLP,
  saves TorchScript and prints MPC vs ANN closed-loop tracking (0.057 m vs 0.061 m RMS cte in simulation).

## Known limitations
- The plant in `sim.py` is kinematic with lags: no tyre slip. Tune again on real FSDS runs before trusting
  the weights at the speed limit (a dynamic model is the next upgrade).
- CMA-ES runs offline in the simulator; a live FSDS evaluation loop (reset service + lap) is not built.
- MPC subscriptions use a reentrant callback group + multi-threaded executor so slow solves cannot starve
  pose/path updates; IPOPT holds the GIL while solving.
- On closed tracks the reference wraps around the path end; on open paths the car slows to `v_end`.
- No stopping logic at the finish (EBS / finished signal) yet.
