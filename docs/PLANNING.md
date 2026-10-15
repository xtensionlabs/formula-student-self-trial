# fs_planning

## Purpose
`path_planner_node` turns the cone map into a drivable centerline and a speed profile.

**Centerline** (`centerline.py`): Delaunay triangulation of yellow+blue cones -> keep edges joining a
yellow to a blue cone (width 2-7 m) -> midpoints -> nearest-neighbour chain that starts just ahead of
the car and never turns back -> light smoothing (removes triangulation zig-zag) -> `CubicSpline`
over arc length (periodic when the chain closes on itself, i.e. the full track is mapped).

**Speed profile** (`speed_gp.py`):
1. curvature `k = |x'y'' - y'x''| / (x'^2+y'^2)^1.5` from the spline;
2. `SpeedGP` (scikit-learn GP, RBF + white noise) gives safe speed for each curvature. It is
   initialised from physics `v = sqrt(a_lat/k)` (a_lat = 15 m/s^2, capped at `v_max`) and the node uses
   `mean - gp_sigma_margin * std` (conservative where the GP is uncertain);
3. forward/backward pass enforces `a_accel` / `a_brake` so the car brakes before corners.

**Online refinement**: a lap is detected when the car returns within `lap_radius` of its start after
`lap_min_length` m. For each curvature bin, evidence from that lap updates the GP: reached the limit
and stayed stable (`|vy/vx| < slip_limit`) -> +`speed_up`; much slipping -> -`speed_down`. Optionally
persisted with `gp_save_path`.

## Subscribed topics
`/slam/map` (until `/slam/map_corrected` arrives, which then takes over), `/slam/pose`, `/state/odometry`.

## Published topics
| Topic | Type |
|---|---|
| `/planning/centerline` | nav_msgs/Path (frame `map`, spacing `cl.ds`) |
| `/planning/velocity_profile` | fs_perception/SpeedProfile (`s`, `speed`, GP `sigma`) |
| `/diagnostics` | planning latency |

## Parameters
`map_topic`, `corrected_topic`, `pose_topic`, `odom_topic`, `plan_hz`, `a_lat_max`, `v_max`, `a_accel`,
`a_brake`, `gp_sigma_margin`, `lap_min_length`, `lap_radius`, `slip_limit`, `speed_up`, `speed_down`,
`gp_save_path`, and all `cl.*` centerline knobs (see `CenterlineParams`).

## Known limitations
- On a closed track the path is one loop starting at the car; the controller must wrap around the end.
- On the first lap the path ends where the map ends; the car must slow down toward the path end.
- Delaunay can create wrong yellow-blue edges at sharp hairpins or after mis-coloured cones; width and
  chain-direction gates reduce but do not remove this.
- The GP refinement is heuristic (per-lap +5% / -15% steps), not full lap-time optimisation.
- Speed limit is purely lateral-acceleration based; no aero/tyre model.
