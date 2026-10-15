"""slam_node: EKF-SLAM on /perception/cones + /state/odometry -> /slam/map, /slam/pose."""
import os

# Small dense matrices: multi-threaded BLAS is ~10x slower here (thread oversubscription).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from typing import Optional
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from fs_perception.msg import ConeArray
from fs_perception_py.diag import LatencyDiag
from fs_slam.ekf_slam import EkfSlam, SlamParams, relative_pose
from fs_slam.msg_utils import cones_to_arrays, map_msg, odom_pose


class SlamNode(Node):
    def __init__(self) -> None:
        super().__init__('slam_node')
        p = self.declare_parameter
        p('cones_topic', '/perception/cones'); p('odom_topic', '/state/odometry')
        p('map_topic', '/slam/map'); p('pose_topic', '/slam/pose'); p('map_frame', 'map')
        d = SlamParams()
        for f in SlamParams.__dataclass_fields__:       # all EKF tuning knobs are parameters
            p(f, getattr(d, f))
        g = lambda n: self.get_parameter(n).value
        self.slam = EkfSlam(SlamParams(**{f: type(getattr(d, f))(g(f)) for f in SlamParams.__dataclass_fields__}))
        self.frame = g('map_frame')
        self.odom: Optional[np.ndarray] = None          # latest odometry pose
        self.last_odom: Optional[np.ndarray] = None     # odometry pose at the previous scan
        self.diag = LatencyDiag(self)
        self.create_subscription(Odometry, g('odom_topic'), self.on_odom, 10)
        self.create_subscription(ConeArray, g('cones_topic'), self.on_cones, 10)
        self.map_pub = self.create_publisher(ConeArray, g('map_topic'), 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, g('pose_topic'), 10)

    def on_odom(self, m: Odometry) -> None:
        self.odom = odom_pose(m)

    def on_cones(self, m: ConeArray) -> None:
        """One SLAM cycle per cone scan: predict from odometry delta, update with cones."""
        if self.odom is None:
            return
        t0 = self.diag.start()
        if self.last_odom is not None:                  # first scan only sets the origin
            self.slam.predict(relative_pose(self.last_odom, self.odom))
        self.last_odom = self.odom.copy()
        xy, col, conf, cov = cones_to_arrays(m)
        self.slam.update(xy, col, conf, cov)
        pos, colors = self.slam.landmarks()
        self.map_pub.publish(map_msg(pos, colors, m.header.stamp, self.frame))
        self._publish_pose(m.header.stamp)
        self.diag.stop(t0)

    def _publish_pose(self, stamp) -> None:
        """Publish the car pose (map frame) with its covariance."""
        x = self.slam.x
        out = PoseWithCovarianceStamped()
        out.header.stamp, out.header.frame_id = stamp, self.frame
        out.pose.pose.position.x, out.pose.pose.position.y = float(x[0]), float(x[1])
        out.pose.pose.orientation.z, out.pose.pose.orientation.w = float(np.sin(x[2] / 2)), float(np.cos(x[2] / 2))
        P = self.slam.P[:3, :3]
        out.pose.covariance[0], out.pose.covariance[7], out.pose.covariance[35] = P[0, 0], P[1, 1], P[2, 2]
        out.pose.covariance[1], out.pose.covariance[6] = P[0, 1], P[1, 0]
        self.pose_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SlamNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
