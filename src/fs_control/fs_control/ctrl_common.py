"""Shared by both controller nodes: pose/path/profile inputs and the (delta, a) <-> command map."""
from typing import Optional, Tuple
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from fs_msgs.msg import ControlCommand
from fs_perception.msg import SpeedProfile
from fs_slam.ekf_slam import relative_pose
from fs_slam.pose_graph import compose
from fs_control.reference import PathRef


def declare_common(node: Node) -> None:
    """Parameters used by both controller nodes."""
    p = node.declare_parameter
    p('pose_topic', '/slam/pose'); p('odom_topic', '/state/odometry')
    p('path_topic', '/planning/centerline'); p('profile_topic', '/planning/velocity_profile')
    p('d_max', 0.436)               # steering angle at |steering| = 1 [rad] (FSDS default 25 deg)
    p('steer_sign', -1.0)           # FSDS: +1 steers RIGHT; our delta > 0 is LEFT (ROS convention)
    p('a_max', 5.0); p('a_min', -10.0)   # accel that maps to full throttle / full brake [m/s^2]


def to_command(node: Node, delta: float, a: float) -> ControlCommand:
    """Steering angle [rad] and acceleration [m/s^2] -> FSDS ControlCommand."""
    g = lambda n: node.get_parameter(n).value
    c = ControlCommand()
    c.header.stamp = node.get_clock().now().to_msg()
    c.steering = float(np.clip(g('steer_sign') * delta / g('d_max'), -1.0, 1.0))
    c.throttle = float(np.clip(a / g('a_max'), 0.0, 1.0))
    c.brake = float(np.clip(a / g('a_min'), 0.0, 1.0)) if a < 0 else 0.0
    return c


def from_command(node: Node, c: ControlCommand) -> Tuple[float, float]:
    """Inverse of `to_command` (used to feed an ANN command back into the MPC bookkeeping)."""
    g = lambda n: node.get_parameter(n).value
    delta = g('steer_sign') * c.steering * g('d_max')
    a = c.throttle * g('a_max') if c.throttle > 0 else -c.brake * abs(g('a_min'))
    return float(delta), float(a)


class ControlInputs:
    """Subscribes to pose, odometry, path and speed profile; exposes the current car state + PathRef.

    `group` should be a reentrant callback group so subscriptions are served while the (slow)
    control timer runs.

    Pose fusion: /slam/pose arrives at scan rate, /state/odometry at 100 Hz. On every SLAM pose we
    store the map<-odom correction T and then pose_now = T (+) odom_now, which is a smooth
    high-rate pose in the SLAM map frame.
    """

    def __init__(self, node: Node, group=None) -> None:
        g = lambda n: node.get_parameter(n).value
        self.T = np.zeros(3)                               # map <- odom correction
        self.have_pose = False
        self.odom_pose, self.v = np.zeros(3), 0.0
        self.ref: Optional[PathRef] = None
        self._path_xy: Optional[np.ndarray] = None
        node.create_subscription(Odometry, g('odom_topic'), self._on_odom, 10, callback_group=group)
        node.create_subscription(PoseWithCovarianceStamped, g('pose_topic'), self._on_pose, 10, callback_group=group)
        node.create_subscription(Path, g('path_topic'), self._on_path, 10, callback_group=group)
        node.create_subscription(SpeedProfile, g('profile_topic'), self._on_profile, 10, callback_group=group)

    def _on_odom(self, m: Odometry) -> None:
        q = m.pose.pose.orientation
        self.odom_pose = np.array([m.pose.pose.position.x, m.pose.pose.position.y,
                                   2 * np.arctan2(q.z, q.w)])
        self.v = m.twist.twist.linear.x

    def _on_pose(self, m: PoseWithCovarianceStamped) -> None:
        q = m.pose.pose.orientation
        slam = np.array([m.pose.pose.position.x, m.pose.pose.position.y, 2 * np.arctan2(q.z, q.w)])
        self.T = compose(slam, relative_pose(self.odom_pose, np.zeros(3)))   # slam (+) inverse(odom)
        self.have_pose = True

    def _on_path(self, m: Path) -> None:
        self._path_xy = np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses])

    def _on_profile(self, m: SpeedProfile) -> None:
        """Combine the newest path with the speed profile (published together by the planner)."""
        if self._path_xy is None or len(self._path_xy) < 5 or not m.s:
            return
        xy = self._path_xy
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        self.ref = PathRef(xy, np.interp(s, m.s, m.speed))

    def state(self) -> Optional[np.ndarray]:
        """Car state [x, y, psi, v] in the map frame, or None until a SLAM pose was received."""
        if not self.have_pose:
            return None
        p = compose(self.T, self.odom_pose)
        return np.array([p[0], p[1], p[2], self.v])
