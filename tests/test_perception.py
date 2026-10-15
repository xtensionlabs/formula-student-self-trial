"""Cone clustering + fusion tests (no ROS needed)."""
import numpy as np
from fs_perception_py.fusion import associate, intrinsics_from_fov, project
from fs_perception_py.lidar_cones import detect_cones

RNG = np.random.default_rng(1)


def ground(n: int = 1500, z: float = -1.0) -> np.ndarray:
    """Flat ground patch 20x20 m, `z` below the sensor."""
    return np.column_stack([RNG.uniform(-10, 10, n), RNG.uniform(-10, 10, n), np.full(n, z) + RNG.normal(0, 0.005, n)])


def cone(x: float, y: float, h: float = 0.32, z0: float = -1.0) -> np.ndarray:
    """A few LiDAR hits on a cone of height h standing at (x, y)."""
    zs = z0 + np.array([0.08, 0.16, 0.24, h - 0.01])
    r = 0.12 * (1 - (zs - z0) / h)             # cone radius shrinks with height
    return np.column_stack([x + r * np.cos(zs * 20), y + r * np.sin(zs * 20), zs])


def test_detects_cones_and_rejects_wall() -> None:
    wall = np.column_stack([np.full(60, 5.0), np.linspace(-1, 1, 60), RNG.uniform(-0.95, -0.4, 60)])
    pts = np.vstack([ground(), cone(4, 2), cone(6, -3), wall])
    cents, covs = detect_cones(pts)
    assert len(cents) == 2 and covs.shape == (2, 4)
    found = sorted(map(tuple, np.round(cents[:, :2]).astype(int)))
    assert found == [(4, 2), (6, -3)]


def test_rejects_too_tall_object() -> None:
    pole = np.column_stack([np.full(8, 3.0), np.full(8, 1.0) + RNG.normal(0, 0.02, 8), np.linspace(-0.95, -0.1, 8)])
    cents, _ = detect_cones(np.vstack([ground(), pole]))
    assert len(cents) == 0                      # 0.85 m tall -> not a cone


def test_projection_and_association() -> None:
    K = intrinsics_from_fov(800, 800, 90)
    # camera frame: x forward, y left. A point 10 m ahead, 2 m to the left -> left of image centre
    uv, valid = project(np.array([[10.0, 2.0, 0.0], [-5.0, 0.0, 0.0]]), K, optical=False)
    assert valid.tolist() == [True, False] and uv[0, 0] < 400 and abs(uv[0, 1] - 400) < 1e-6
    # same points in an optical frame (x right, y down, z forward) must land on the same pixels
    uv_opt, _ = project(np.array([[-2.0, 0.0, 10.0]]), K)
    assert np.allclose(uv_opt[0], uv[0])
    boxes = np.array([[uv[0, 0] - 10, 380, uv[0, 0] + 10, 420], [700, 700, 760, 760]])
    assert associate(uv, valid, boxes, gate_px=50).tolist() == [0, -1]   # behind-camera cone unmatched
