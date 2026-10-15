"""cone_detector_node: LiDAR cone detection + YOLO colour + frustum fusion -> /perception/cones.

Image callback runs YOLO and caches the latest boxes; the LiDAR callback detects cones,
projects them into the cached image and takes the colour of the matching box.
Without a YOLO model the node still works LiDAR-only (all cones colour 0 = unknown).
"""
from typing import Optional
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_ros import Buffer, TransformListener, TransformException
from fs_perception.msg import Cone, ConeArray
from fs_perception_py.diag import LatencyDiag
from fs_perception_py.lidar_cones import LidarParams, detect_cones
from fs_perception_py.fusion import associate, intrinsics_from_fov, project, quat_to_matrix

COLOR_IDS = {'yellow': 1, 'blue': 2, 'orange': 3}   # class-name keyword -> Cone.color


def cloud_to_xyz(m: PointCloud2) -> np.ndarray:
    """Decode x/y/z float32 fields of a PointCloud2 into an (N,3) array."""
    off = {f.name: f.offset for f in m.fields}
    dt = np.dtype({'names': ['x', 'y', 'z'], 'formats': ['<f4'] * 3,
                   'offsets': [off['x'], off['y'], off['z']], 'itemsize': m.point_step})
    a = np.frombuffer(bytes(m.data), dtype=dt)
    return np.stack([a['x'], a['y'], a['z']], axis=1).astype(np.float64)


class ConeDetector(Node):
    def __init__(self) -> None:
        super().__init__('cone_detector_node')
        p = self.declare_parameter
        p('lidar_topic', '/lidar/pointcloud2'); p('image_topic', '/camera/image_color')
        p('camera_info_topic', '/camera/camera_info'); p('cones_topic', '/perception/cones')
        p('camera_frame', 'fsds/cam1')
        p('vehicle_frame', 'fsds/FSCar')           # output frame of the cones
        p('use_camera', True)                 # False: LiDAR-only instance (e.g. the second LiDAR), no image processing
        p('model_path', ''); p('yolo_conf', 0.4)   # YOLOv8 weights ('' = LiDAR only)
        p('camera_optical', True)             # camera TF frame is optical (FSDS bridge) rather than ENU
        p('gate_px', 60.0)                         # max pixel distance for LiDAR<->box match
        p('image_max_age', 0.3)                    # s; older detections are not used
        p('cam_width', 785); p('cam_height', 785); p('cam_fov', 90.0)   # fallback intrinsics
        defaults = LidarParams()
        for f in LidarParams.__dataclass_fields__:  # every LiDAR tuning knob is a parameter
            p('lidar.' + f, getattr(defaults, f))
        g = lambda n: self.get_parameter(n).value
        self.lp = LidarParams(**{f: type(getattr(defaults, f))(g('lidar.' + f))
                                 for f in LidarParams.__dataclass_fields__})
        self.K = intrinsics_from_fov(g('cam_width'), g('cam_height'), g('cam_fov'))
        self.yolo = self._load_yolo(g('model_path')) if g('use_camera') else None
        self.det: Optional[dict] = None            # latest camera detections
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.diag = LatencyDiag(self)
        self.create_subscription(PointCloud2, g('lidar_topic'), self.on_lidar, qos_profile_sensor_data)
        if g('use_camera'):
            self.create_subscription(Image, g('image_topic'), self.on_image, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, g('camera_info_topic'), self.on_info, 1)
        self.pub = self.create_publisher(ConeArray, g('cones_topic'), 10)

    def _load_yolo(self, path: str):
        """Load YOLOv8 weights; return None (LiDAR-only mode) if unavailable."""
        if not path:
            self.get_logger().warn('no model_path set: running LiDAR-only, cone colours unknown')
            return None
        from ultralytics import YOLO                 # imported lazily: heavy import
        model = YOLO(path)
        self.color_of = {i: next((c for k, c in COLOR_IDS.items() if k in n.lower()), 0)
                         for i, n in model.names.items()}
        self.get_logger().info(f'YOLO loaded: {path} classes={model.names}')
        return model

    def on_info(self, m: CameraInfo) -> None:
        """Prefer real intrinsics when the camera publishes them."""
        self.K = np.array(m.k).reshape(3, 3)

    def on_image(self, m: Image) -> None:
        """Run YOLO on the newest image and cache boxes/colours/confidences."""
        if self.yolo is None:
            return
        t0 = self.diag.start()
        img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)[:, :, :3]
        if m.encoding.startswith('rgb'):
            img = img[:, :, ::-1]                    # ultralytics expects BGR arrays
        r = self.yolo.predict(np.ascontiguousarray(img), conf=self.get_parameter('yolo_conf').value,
                              verbose=False)[0]
        self.det = {'t': Time.from_msg(m.header.stamp).nanoseconds * 1e-9,
                    'boxes': r.boxes.xyxy.cpu().numpy(), 'conf': r.boxes.conf.cpu().numpy(),
                    'color': np.array([self.color_of[int(c)] for c in r.boxes.cls.cpu().numpy()], dtype=int)}
        self.diag.stop(t0)

    def _tf_matrix(self, target: str, source: str) -> Optional[np.ndarray]:
        """4x4 transform taking points from `source` frame into `target` frame (latest TF)."""
        try:
            t = self.tf.lookup_transform(target, source, Time()).transform
        except TransformException as e:
            self.get_logger().warn(f'TF {source}->{target} unavailable: {e}', throttle_duration_sec=5.0)
            return None
        T = np.eye(4)
        T[:3, :3] = quat_to_matrix(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w)
        T[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
        return T

    def on_lidar(self, m: PointCloud2) -> None:
        t0 = self.diag.start()
        cents, covs = detect_cones(cloud_to_xyz(m), self.lp)
        colors, confs = np.zeros(len(cents), dtype=int), np.full(len(cents), 0.5)
        g = lambda n: self.get_parameter(n).value
        T_cam = self._tf_matrix(g('camera_frame'), m.header.frame_id) if len(cents) else None
        stamp = Time.from_msg(m.header.stamp).nanoseconds * 1e-9
        if (T_cam is not None and self.det is not None and len(self.det['boxes'])
                and abs(stamp - self.det['t']) < g('image_max_age')):
            # Frustum fusion: LiDAR centroids -> camera pixels -> nearest detection box
            cam = cents @ T_cam[:3, :3].T + T_cam[:3, 3]
            uv, valid = project(cam, self.K, g('camera_optical'))
            match = associate(uv, valid, self.det['boxes'], g('gate_px'))
            hit = match >= 0
            colors[hit] = self.det['color'][match[hit]]
            confs[hit] = self.det['conf'][match[hit]]
        # Express cones in the vehicle frame for SLAM
        T_veh = self._tf_matrix(g('vehicle_frame'), m.header.frame_id)
        out = ConeArray()
        out.header.stamp, out.header.frame_id = m.header.stamp, g('vehicle_frame')
        if T_veh is not None and len(cents):
            veh = cents @ T_veh[:3, :3].T + T_veh[:3, 3]
            R2 = T_veh[:2, :2]
            for pt, cv, col, cf in zip(veh, covs, colors, confs):
                c = Cone(x=float(pt[0]), y=float(pt[1]), z=float(pt[2]), color=int(col),
                         confidence=float(cf))
                c.covariance = (R2 @ cv.reshape(2, 2) @ R2.T).ravel().astype(np.float32).tolist()
                out.cones.append(c)
        self.pub.publish(out)
        self.diag.stop(t0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConeDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
