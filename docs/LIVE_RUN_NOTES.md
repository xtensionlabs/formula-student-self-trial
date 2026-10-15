# Live simulator runs - findings and how to run (2026-10-15)

## Running the sim on this laptop (Ryzen 5 7530U, integrated GPU)
Start FSDS low-res/low-quality or the physics runs at ~1/3 real time and every dead-reckoning estimate is 2-3x off:

```
FSDS.exe -windowed -ResX=480 -ResY=270 -nosound -ExecCmds="r.ScreenPercentage 50,sg.ViewDistanceQuality 0,sg.ShadowQuality 0,sg.PostProcessQuality 0,sg.TextureQuality 0,sg.EffectsQuality 0,sg.FoliageQuality 0,r.VSync 0,t.MaxFPS 120"
```
Check real-time factor: `scripts/gt_pursuit.py` prints distance / stamp-time; it must equal the reported speed.
Then `scripts/run_bridge.sh` (restarts the bridge when the sim RPC times out), then `ros2 launch fs_bringup autonomous.launch.py`.
The real process is `Blocks.exe`; restart the sim after editing `settings.json`.

## Findings
| Finding | Evidence | Status |
|---|---|---|
| Sim physics slower than stamps | GT speed 3.05 m/s vs 1.01 m/s distance per stamp-second | fixed by low-res start (ratio 1.0) |
| Cameras choke the iGPU, bridge dies on RPC timeout | `simGetImages` timeouts | cameras removed from settings |
| 4-laser LiDAR misses cones | recall 33% at 10-15 m | 8 lasers, -0.5..-4 deg: 100% recall 5-20 m, precision 100%, 4 cm |
| MPC infeasible at standstill (v = -0.0) | `Maximum_CpuTime_Exceeded` on every solve | speed clamped to >= 0 |
| Planner burned 6 cores (threaded BLAS) | top: 630% CPU | single-thread BLAS env in launch |
| MPC too slow under load | 50-80 ms | CasADi `expand`, tol 1e-2: ~11 ms |
| Steering sign/gain | pure pursuit on ground truth: mean off 0.17 m | OK (+steering = right) |
| Full-stack drive still leaves the track | monitor_drive logs | open: verify pose + planned path vs ground truth |

## Direction from the user
Completing the track with few cone hits matters more than speed (target ~1 m/s first). Suspected root cause is
mapping / path planning / perception, not control. Add recovery: keep re-planning a path after cone hits.

## Tools (scripts/)
`monitor_drive.py` live log + CSV, `eval_perception.py`, `diag_planning.py`, `mpc_probe.py`, `gt_pursuit.py [--compare]`,
`clock_check.py`, `lidar_probe.py`, `check_tf.py`, `sim_tools.py reset|steer`, `stack_check.py`, `list_topics.py`.
Quirks: ros2 CLI is very slow (use these scripts); Fast DDS needs a third participant to discover the bridge
(keep the stack or a dummy node running); UDP-only DDS does not work here.
