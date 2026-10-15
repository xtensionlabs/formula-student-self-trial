"""loop_closure_node: keyframe pose graph + cone-based loop closure -> /slam/map_corrected."""
import os

# Small dense matrices: multi-threaded BLAS is ~10x slower here (thread oversubscription).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from typing import Optional
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from nav_msgs.msg import Odometry
from fs_perception.msg import ConeArray
from fs_perception_py.diag import LatencyDiag
from fs_slam.ekf_slam import relative_pose
from fs_slam.msg_utils import cones_to_arrays, map_msg, odom_pose
from fs_slam.pose_graph import GraphParams, PoseGraph


class LoopClosureNode(Node):
    def __init__(self) -> None:
        super().__init__('loop_closure_node')
        p = self.declare_parameter
        p('cones_topic', '/perception/cones'); p('odom_topic', '/state/odometry')
        p('corrected_topic', '/slam/map_corrected'); p('map_frame', 'map')
        d = GraphParams()
        for f in GraphParams.__dataclass_fields__:
            p(f, getattr(d, f))
        g = lambda n: self.get_parameter(n).value
        self.graph = PoseGraph(GraphParams(**{f: type(getattr(d, f))(g(f)) for f in GraphParams.__dataclass_fields__}))
        self.frame = g('map_frame')
        self.origin: Optional[np.ndarray] = None        # first odometry pose = map-frame origin
        self.odom: Optional[np.ndarray] = None
        self.diag = LatencyDiag(self)
        self.create_subscription(Odometry, g('odom_topic'), lambda m: setattr(self, 'odom', odom_pose(m)), 10)
        self.create_subscription(ConeArray, g('cones_topic'), self.on_cones, 10)
        # Transient-local so a late planner still receives the latest corrected map
        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(ConeArray, g('corrected_topic'), qos)

    def on_cones(self, m: ConeArray) -> None:
        if self.odom is None:
            return
        if self.origin is None:
            self.origin = self.odom.copy()
        pose = relative_pose(self.origin, self.odom)    # odometry pose in the map frame
        if not self.graph.needs_keyframe(pose):
            return
        t0 = self.diag.start()
        xy, col, _, _ = cones_to_arrays(m)
        closed = self.graph.add_keyframe(pose, xy, col)
        if closed:
            self.get_logger().info(f'loop closure #{self.graph.n_loops} at keyframe {len(self.graph.kfs) - 1}')
        if self.graph.n_loops:                          # once a loop exists, keep publishing the corrected map
            pos, colors = self.graph.cone_map()
            self.pub.publish(map_msg(pos, colors, m.header.stamp, self.frame))
        self.diag.stop(t0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LoopClosureNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
