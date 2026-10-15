"""Tiny synthetic circular track + noisy odometry/cone simulator for tests (no ROS)."""
from typing import Iterator, Tuple
import numpy as np

R, HALF_W, SPACING = 15.0, 1.75, 3.0       # centreline radius, half track width, cone spacing [m]


def make_cones() -> Tuple[np.ndarray, np.ndarray]:
    """Cones in the world frame. Driving counter-clockwise: blue (2) inside/left, yellow (1) outside/right."""
    pts, col = [], []
    for r, c in ((R - HALF_W, 2), (R + HALF_W, 1)):
        n = int(2 * np.pi * r / SPACING)
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
        col.append(np.full(n, c))
    return np.vstack(pts), np.concatenate(col)


def to_frame(pose: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """World points -> car frame of `pose`."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return (pts - pose[:2]) @ np.array([[c, -s], [s, c]])


def drive(laps: float = 1.2, step: float = 0.5, odom_bias: float = 1.005, yaw_bias: float = 0.0003,
          seed: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (true_pose, odom_pose, cone_xy, cone_colour, cone_cov) per step.

    Odometry integrates biased + noisy deltas, so it drifts; cone observations are noisy,
    limited to 12 m range and +-60 deg field of view, and expressed in the car frame.
    """
    rng = np.random.default_rng(seed)
    cones, colors = make_cones()
    odom = np.zeros(3)
    prev = None
    for k in range(int(laps * 2 * np.pi * R / step)):
        th = k * step / R
        pose = np.array([R * np.cos(th), R * np.sin(th), th + np.pi / 2])
        if prev is not None:
            c, s = np.cos(prev[2]), np.sin(prev[2])
            d = pose[:2] - prev[:2]
            local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
            dyaw = pose[2] - prev[2]
            local = local * odom_bias + rng.normal(0, 0.01, 2)           # scale drift + noise
            dyaw = dyaw + yaw_bias + rng.normal(0, 0.002)               # heading drift + noise
            co, so = np.cos(odom[2]), np.sin(odom[2])
            odom = np.array([odom[0] + co * local[0] - so * local[1],
                             odom[1] + so * local[0] + co * local[1], odom[2] + dyaw])
        prev = pose
        rel = to_frame(pose, cones)
        rng_b = np.hypot(rel[:, 0], rel[:, 1])
        vis = (rng_b < 12) & (rel[:, 0] > 0) & (np.abs(np.arctan2(rel[:, 1], rel[:, 0])) < np.radians(60))
        xy = rel[vis] + rng.normal(0, 0.05, (vis.sum(), 2))
        cov = np.tile(np.eye(2) * 0.05 ** 2, (vis.sum(), 1, 1))
        yield pose, odom.copy(), xy, colors[vis], cov
