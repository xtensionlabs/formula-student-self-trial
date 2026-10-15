"""LiDAR cone detection (pure numpy/scipy, no ROS): ground removal -> clustering -> cone filter.

Input points are in the LiDAR frame (z up). Output cones are (N,3) centroids plus (N,4)
row-major 2x2 xy covariances.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass
class LidarParams:
    max_range: float = 25.0        # ignore points further than this [m] (xy distance)
    ground_dist: float = 0.05      # RANSAC inlier distance to the ground plane [m]
    ransac_iters: int = 60
    max_height: float = 0.7        # discard points higher above ground than this [m]
    cluster_eps: float = 0.3       # max neighbour distance inside one cluster [m]
    min_points: int = 3
    h_min: float = 0.2             # cone top height above ground [m] (spec: 0.2-0.5)
    h_max: float = 0.55            # (large orange cones reach ~0.5)
    max_footprint: float = 0.45    # max xy extent of a cone cluster [m]
    min_aspect: float = 0.5        # height / footprint, rejects flat objects
    sigma0: float = 0.05           # position std at range 0 [m]
    sigma_per_m: float = 0.004     # extra std per metre of range


def fit_ground(pts: np.ndarray, p: LidarParams, rng: np.random.Generator) -> Optional[np.ndarray]:
    """RANSAC plane fit. Returns (a,b,c,d) with a*x+b*y+c*z+d=0 and unit normal, or None."""
    if len(pts) < 3:
        return None
    sample = pts if len(pts) <= 4000 else pts[rng.choice(len(pts), 4000, replace=False)]
    best, best_n = None, 0
    for _ in range(p.ransac_iters):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if abs(n[2]) < 0.9:                      # ground must be near-horizontal
            continue
        if n[2] < 0:
            n = -n                               # normal points up
        d = -n @ a
        count = int(np.sum(np.abs(sample @ n + d) < p.ground_dist))
        if count > best_n:
            best, best_n = np.append(n, d), count
    return best


def detect_cones(pts: np.ndarray, p: LidarParams = LidarParams(),
                 seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (centroids (N,3), covariances (N,4)) of cone-like clusters."""
    empty = np.zeros((0, 3)), np.zeros((0, 4))
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < p.max_range]      # range crop
    plane = fit_ground(pts, p, np.random.default_rng(seed))
    if plane is None:
        return empty
    h = pts @ plane[:3] + plane[3]                               # signed height above ground
    keep = (h > p.ground_dist) & (h < p.max_height)              # drop ground + tall stuff
    pts, h = pts[keep], h[keep]
    if len(pts) < p.min_points:
        return empty

    # Euclidean clustering: connect points closer than eps, take connected components
    pairs = cKDTree(pts).query_pairs(p.cluster_eps, output_type='ndarray')
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(pts),) * 2)
    _, labels = connected_components(g, directed=False)

    cents, covs = [], []
    for lab in np.unique(labels):
        idx = labels == lab
        if idx.sum() < p.min_points:
            continue
        c, hc = pts[idx], h[idx]
        foot = max(np.ptp(c[:, 0]), np.ptp(c[:, 1]))             # xy extent of the cluster
        top = hc.max()                                           # cone height = top above ground
        if not (p.h_min <= top <= p.h_max) or foot > p.max_footprint:
            continue
        if top / max(foot, 0.05) < p.min_aspect:                 # too flat/wide -> not a cone
            continue
        cen = c.mean(axis=0)
        # Position covariance: grows with range, elongated along the viewing ray
        r = np.hypot(cen[0], cen[1])
        sr = p.sigma0 + p.sigma_per_m * r
        st = p.sigma0 + 2 * p.sigma_per_m * r
        u = np.array([cen[0], cen[1]]) / max(r, 1e-6)            # unit ray direction
        cov = sr ** 2 * np.outer(u, u) + st ** 2 * (np.eye(2) - np.outer(u, u))
        cents.append(cen)
        covs.append(cov.ravel())
    if not cents:
        return empty
    return np.array(cents), np.array(covs)
