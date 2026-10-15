# fs_perception

## Purpose
`cone_detector_node` turns LiDAR + camera data into a coloured cone list in the vehicle frame.
The package also owns the stack's custom messages (`Cone`, `ConeArray`, `SpeedProfile`);
its Python code lives in `fs_perception_py`.

Pipeline
1. `lidar_cones.py` (numpy/scipy, no ROS): range crop -> **RANSAC ground plane** -> drop ground
   and tall points -> **Euclidean clustering** (KD-tree pairs + connected components) -> cone
   filter (top height 0.2-0.55 m above ground, xy footprint <= 0.45 m, height/footprint >= 0.5)
   -> centroid + 2x2 covariance (grows with range, elongated along the viewing ray).
2. Camera: **YOLOv8** (ultralytics) boxes + class + confidence, cached for the next LiDAR scan.
3. `fusion.py`: project centroids with intrinsics (from `camera_info`, else FOV params) and TF
   extrinsics, then **optimal one-to-one assignment** (Hungarian) to box centres within `gate_px`.
   Unmatched cones stay colour 0 (unknown).

No `open3d`: RANSAC + clustering on a sparse FSDS cloud is ~30 lines of numpy, faster to debug.

## Subscribed topics (names are parameters; FSDS uses `/fsds/...` prefixes)
| Topic | Type |
|---|---|
| `/lidar/pointcloud2` | sensor_msgs/PointCloud2 (sensor frame, e.g. `fsds/Lidar1`) |
| `/camera/image_color` | sensor_msgs/Image (rgb8/bgr8) |
| `/camera/camera_info` | sensor_msgs/CameraInfo (optional) |
| `/tf_static` | extrinsics lidar -> camera, lidar -> vehicle |

## Published topics
| Topic | Type |
|---|---|
| `/perception/cones` | fs_perception/ConeArray (frame `vehicle_frame`) |
| `/diagnostics` | latency of LiDAR and YOLO callbacks |

## Parameters
`lidar_topic`, `image_topic`, `camera_info_topic`, `cones_topic`, `camera_frame`, `vehicle_frame`,
`model_path` (empty = LiDAR-only), `yolo_conf`, `gate_px`, `image_max_age`,
`cam_width/height/fov`, and all `lidar.*` knobs (see `LidarParams` in `lidar_cones.py`).

## Known limitations
- The bridge publishes no `camera_info`; intrinsics fall back to the FOV parameters.
- Camera frames are ENU (x forward); `fusion.project` converts to optical axes. Verify the
  `camera_frame` name against your `settings.json` camera name.
- YOLO weights must be fine-tuned on FSDS cones (`models/cones.pt`, not trained yet).
- LiDAR and image are fused by latest-image (age-gated), not by exact timestamp.
- Ground model is one plane; strong slopes/banking would need a piecewise model.

## Measured on FSDS (scripts/eval_perception.py, car standing at the start, ground truth from the sim)
| | Lidar1 only | Lidar1 + Lidar2 |
|---|---|---|
| precision | 100 % | 100 % |
| position error (mean / p95) | 5.0 / 5.8 cm | 4.7 / 6.8 cm |
| recall 5-10 m / 10-15 m / 15-20 m | 67 / 33 / 0 % | 78 / 67 / 67 % |
| recall 0-5 m | 0 % | 0 % (open) |

Why: Lidar1 has 4 lasers but only the -5 and -2 degree rings return, so a 0.33 m cone is hit by one ring and only
between ~2.6 m and ~15.8 m. Lidar2 (0.2 m high) hits cones at ~1.4-2.3 m and ~3.6-5.7 m but only 2-7 cm above the
ground, which is the same size as the ground threshold. Next: test 1 cm ground threshold (needs a fresh sim run),
or use a denser LiDAR in settings.json (the cheapest big win).
The bridge camera frame is an optical frame (x right, y down, z forward): parameter camera_optical (default true).
Tools: scripts/lidar_probe.py (ring structure + ground height), scripts/check_tf.py, scripts/eval_perception.py.
