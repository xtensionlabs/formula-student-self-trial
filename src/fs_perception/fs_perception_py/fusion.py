"""Camera-LiDAR fusion helpers (pure numpy, no ROS)."""
from typing import Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

UNKNOWN = 0


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """3x3 rotation matrix from a quaternion."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def intrinsics_from_fov(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Pinhole K from horizontal FOV (the bridge publishes no camera_info)."""
    f = width / (2 * np.tan(np.radians(fov_deg) / 2))
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def project(pts_cam: np.ndarray, K: np.ndarray, optical: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Project points in the camera frame to pixels.

    optical=True: frame is already optical (x right, y down, z forward) - what the FSDS bridge publishes
    (verified from the TF of fsds/cam1). optical=False: ENU-style (x fwd, y left, z up).

    Returns (uv (N,2), valid (N,) bool) - valid means the point is in front of the camera.
    """
    opt = pts_cam if optical else np.stack([-pts_cam[:, 1], -pts_cam[:, 2], pts_cam[:, 0]], axis=1)
    valid = opt[:, 2] > 0.1
    uvw = opt @ K.T
    uv = uvw[:, :2] / np.where(valid, uvw[:, 2], 1.0)[:, None]
    return uv, valid


def associate(uv: np.ndarray, valid: np.ndarray, boxes: np.ndarray, gate_px: float) -> np.ndarray:
    """One-to-one match of projected cones to detection boxes by box-centre distance.

    boxes: (M,4) x1,y1,x2,y2. Returns box index per cone, -1 if unmatched (or beyond gate).
    """
    match = -np.ones(len(uv), dtype=int)
    if len(uv) == 0 or len(boxes) == 0:
        return match
    centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    cost = np.linalg.norm(uv[:, None, :] - centres[None, :, :], axis=2)
    cost[~valid] = 1e6                                   # cones behind the camera never match
    rows, cols = linear_sum_assignment(cost)             # optimal one-to-one assignment
    for r, c in zip(rows, cols):
        if cost[r, c] <= gate_px:
            match[r] = c
    return match
