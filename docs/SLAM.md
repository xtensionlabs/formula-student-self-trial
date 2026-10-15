# fs_slam

## Purpose
Two nodes with separate jobs:

- **`slam_node`** – online EKF-SLAM (`ekf_slam.py`). State `[car pose | cone_1 | ... | cone_N]` in
  the map frame (origin = first odometry pose). Motion model = odometry delta from
  `/state/odometry`; observation = range + bearing of each cone. Association = nearest neighbour with
  Mahalanobis gating (chi2 99%); observations between the match gate and the new-landmark gate are
  ambiguous and ignored. Colours are tracked per landmark by confidence-weighted votes and cones only
  associate with landmarks of a compatible colour, so yellow and blue form separate maps. A landmark
  is published after `confirm_hits` observations (rejects LiDAR false positives).
- **`loop_closure_node`** – keyframe pose graph (`pose_graph.py`, scipy only; no g2o/GTSAM needed).
  Keyframes every `kf_dist` m / `kf_yaw` rad store odometry pose + cones. A keyframe closes a loop when
  an older one (>= `loop_min_gap` keyframes back, within `loop_radius`) aligns by 2D ICP on cones
  (colour-consistent, coarse-to-fine gate), the correction is plausible (`max_shift`, `max_turn`) and
  inliers/RMS pass. The SE(2) graph is then re-optimised (least squares) and the cones of all keyframes
  are merged into a corrected map.

## Subscribed topics
| Topic | Type | Used by |
|---|---|---|
| `/perception/cones` | fs_perception/ConeArray | both |
| `/state/odometry` | nav_msgs/Odometry | both |

## Published topics
| Topic | Type | Node |
|---|---|---|
| `/slam/map` | ConeArray (frame `map`) | slam_node |
| `/slam/pose` | PoseWithCovarianceStamped | slam_node |
| `/slam/map_corrected` | ConeArray, transient-local | loop_closure_node (only after a loop closure) |
| `/diagnostics` | latency per cycle | both |

## Parameters
slam_node: `cones_topic`, `odom_topic`, `map_topic`, `pose_topic`, `map_frame` plus every field of
`SlamParams` (odometry noise, sensor-noise floors, `match_gate`, `new_gate`, `search_radius`,
`confirm_hits`). loop_closure_node: topics + every field of `GraphParams` (`kf_dist`, `loop_min_gap`,
`loop_radius`, ICP and plausibility gates, `merge_eps`, `min_obs`).

## Known limitations
- Covariance is dense (O(N^2)); fine for ~300 cones, revisit for much bigger maps.
- The EKF does not consume the corrected map: planning should use `/slam/map_corrected` once it exists.
- Plausibility gate `max_shift` must stay below half the cone spacing, otherwise ICP can lock onto a
  neighbouring cone (periodic cone rows); odometry drift per lap must stay below it.
- Odometry and cones are paired by "latest message", not interpolated by timestamp.
- Spurious unconfirmed landmarks are kept in the EKF state (never pruned).
- Both nodes pin BLAS to one thread (multi-threaded BLAS was ~10x slower on these small matrices).
- Tests use a synthetic circular track (`tests/track_sim.py`), not the real simulator yet.
