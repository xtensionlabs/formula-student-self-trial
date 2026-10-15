# fs_state_estimation

## Purpose
`state_estimator_node` fuses IMU and ground-speed-sensor data in an Unscented Kalman Filter
(`ukf.py`, ROS-free) and publishes a smooth 100 Hz odometry estimate.

State `[x, y, yaw, vx, vy, yaw_rate]`; position/heading in the odom frame (origin = start pose),
velocities in the body frame.

**Process model** – kinematic bicycle: pose integrates body velocities, `vx` follows IMU `ax`,
`vy = lr * yaw_rate` (no-slip), yaw rate is a random walk.
**Updates** – IMU yaw + yaw rate; GSS velocity (world or body frame); a weak pseudo-measurement
`vy - lr*r = 0` after each IMU step.

## Subscribed topics
| Topic | Type |
|---|---|
| `/imu` (param `imu_topic`) | sensor_msgs/Imu |
| `/gss` (param `gss_topic`) | geometry_msgs/TwistWithCovarianceStamped |

## Published topics
| Topic | Type |
|---|---|
| `/state/odometry` | nav_msgs/Odometry (100 Hz, frame `odom`→`base_link`) |
| `/diagnostics` | diagnostic_msgs/DiagnosticArray (`latency_ms` of the IMU callback) |

## Parameters
`imu_topic`, `gss_topic`, `odom_topic`, `gss_frame` (`auto`|`world`|`body`), `rate_hz`, `lr`,
`q_diag` (process noise), `r_imu`, `r_gss`, `r_bicycle`.

## Known limitations
- FSDS bridge topics are namespaced (`/fsds/imu`, `/fsds/gss`); remap via parameters/launch.
- The GSS frame is not documented by FSDS: `gss_frame: auto` (default) detects world vs body from the first turn (needs speed > 1 m/s and |yaw| > 0.15 rad); until then only standstill updates are applied. Check the "GSS frame detected" log line.
- No wheel odometry yet and no bias estimation; position drifts (SLAM corrects it).
- Process noise values are initial guesses; tune on recorded data.
