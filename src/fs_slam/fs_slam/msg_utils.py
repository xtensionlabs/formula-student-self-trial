"""Small ROS-message <-> numpy converters shared by the SLAM nodes."""
from typing import Tuple
import numpy as np
from nav_msgs.msg import Odometry
from fs_perception.msg import Cone, ConeArray


def odom_pose(m: Odometry) -> np.ndarray:
    """(x, y, yaw) from an Odometry message."""
    q = m.pose.pose.orientation
    yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return np.array([m.pose.pose.position.x, m.pose.pose.position.y, yaw])


def cones_to_arrays(m: ConeArray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(xy (M,2), colours (M,), confidence (M,), 2x2 covariances (M,2,2))."""
    xy = np.array([[c.x, c.y] for c in m.cones]).reshape(-1, 2)
    col = np.array([c.color for c in m.cones], dtype=int)
    conf = np.array([c.confidence for c in m.cones])
    cov = np.array([c.covariance for c in m.cones]).reshape(-1, 2, 2)
    return xy, col, conf, cov


def map_msg(pos: np.ndarray, col: np.ndarray, stamp, frame: str) -> ConeArray:
    """Build a ConeArray from map positions and colours."""
    out = ConeArray()
    out.header.stamp, out.header.frame_id = stamp, frame
    out.cones = [Cone(x=float(p[0]), y=float(p[1]), color=int(c), confidence=1.0)
                 for p, c in zip(pos, col)]
    return out
