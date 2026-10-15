"""Centerline from a cone map (ROS-free): Delaunay -> yellow/blue edge midpoints -> chain -> spline."""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import Delaunay, QhullError

YELLOW, BLUE = 1, 2


@dataclass
class CenterlineParams:
    min_width: float = 2.0      # accepted yellow<->blue edge length [m]
    max_width: float = 7.0
    max_gap: float = 6.0        # max distance between consecutive waypoints in the chain [m]
    start_ahead: float = 12.0   # chain starts at the nearest waypoint within this distance ahead [m]
    close_dist: float = 4.0     # chain end this close to its start (and long enough) = closed track
    min_closed_pts: int = 20
    smooth_passes: int = 2      # 3-tap moving-average passes that remove Delaunay zig-zag
    ds: float = 0.5             # spline sampling distance [m]


def midpoints(pos: np.ndarray, col: np.ndarray, p: CenterlineParams) -> np.ndarray:
    """Midpoints of Delaunay edges that join a yellow cone to a blue cone."""
    keep = np.isin(col, (YELLOW, BLUE))
    pos, col = pos[keep], col[keep]
    if len(pos) < 3 or not ((col == YELLOW).any() and (col == BLUE).any()):
        return np.zeros((0, 2))
    try:
        tri = Delaunay(pos)
    except QhullError:                                   # collinear / degenerate input
        return np.zeros((0, 2))
    edges = set()
    for a, b, c in tri.simplices:                        # collect each undirected edge once
        edges.update({tuple(sorted(e)) for e in ((a, b), (b, c), (a, c))})
    mids = []
    for i, j in edges:
        if col[i] != col[j]:                             # yellow-blue edge crosses the track
            w = np.linalg.norm(pos[i] - pos[j])
            if p.min_width <= w <= p.max_width:
                mids.append((pos[i] + pos[j]) / 2)
    return np.array(mids).reshape(-1, 2)


def order_chain(wp: np.ndarray, car: np.ndarray, p: CenterlineParams) -> np.ndarray:
    """Nearest-neighbour chain over waypoints, starting just ahead of the car, never turning back."""
    if len(wp) == 0:
        return wp
    heading = np.array([np.cos(car[2]), np.sin(car[2])])
    rel = wp - car[:2]
    dist = np.linalg.norm(rel, axis=1)
    ahead = (rel @ heading > 0) & (dist < p.start_ahead)
    start = int(np.argmin(np.where(ahead, dist, np.inf))) if ahead.any() else int(np.argmin(dist))
    chain, left, direction = [start], set(range(len(wp))) - {start}, heading
    while left:
        cur = wp[chain[-1]]
        idx = np.array(sorted(left))
        step = wp[idx] - cur
        d = np.linalg.norm(step, axis=1)
        ok = (d < p.max_gap) & (step @ direction > 0)   # close enough and not behind us
        if not ok.any():
            break
        nxt = idx[np.argmin(np.where(ok, d, np.inf))]
        direction = (wp[nxt] - cur) / max(np.linalg.norm(wp[nxt] - cur), 1e-9)
        chain.append(int(nxt))
        left.discard(int(nxt))
    return wp[chain]


def smooth(wp: np.ndarray, closed: bool, passes: int) -> np.ndarray:
    """3-tap moving average; ends stay fixed on open chains."""
    for _ in range(passes):
        if len(wp) < 3:
            break
        new = wp.copy()
        if closed:
            new = (np.roll(wp, 1, 0) + wp + np.roll(wp, -1, 0)) / 3
        else:
            new[1:-1] = (wp[:-2] + wp[1:-1] + wp[2:]) / 3
        wp = new
    return wp


def build_centerline(pos: np.ndarray, col: np.ndarray, car: np.ndarray,
                     p: CenterlineParams = CenterlineParams()) -> Optional[Tuple[CubicSpline, float, bool]]:
    """Return (spline over arc length, total length, closed) or None if no path can be built."""
    chain = order_chain(midpoints(pos, col, p), car, p)
    if len(chain) < 4:
        return None
    closed = len(chain) >= p.min_closed_pts and np.linalg.norm(chain[-1] - chain[0]) < p.close_dist
    chain = smooth(chain, closed, p.smooth_passes)
    if closed:
        chain = np.vstack([chain, chain[0]])             # repeat first point for the periodic spline
    seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    keep = np.r_[True, seg > 0.05]                       # drop duplicates: spline needs increasing s
    chain = chain[keep]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(chain, axis=0), axis=1))]
    if len(chain) < 4:
        return None
    return CubicSpline(s, chain, bc_type='periodic' if closed else 'not-a-knot'), float(s[-1]), bool(closed)


def sample(spline: CubicSpline, length: float, ds: float, closed: bool
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample the spline: (s, xy, curvature, heading)."""
    s = np.arange(0.0, length, ds) if closed else np.arange(0.0, length + 1e-9, ds)
    xy, d1, d2 = spline(s), spline(s, 1), spline(s, 2)
    # kappa = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    kappa = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum((d1 ** 2).sum(1), 1e-9) ** 1.5
    return s, xy, kappa, np.arctan2(d1[:, 1], d1[:, 0])
