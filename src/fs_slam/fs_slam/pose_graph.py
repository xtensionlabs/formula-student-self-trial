"""Keyframe pose graph with cone-based loop closure (ROS-free, scipy only).

Keyframes hold the odometry pose and the cones seen from that pose (car frame).
Odometry edges link consecutive keyframes; a loop edge is added when the cones of a new
keyframe can be aligned (2D ICP) with those of an old keyframe that is spatially close but
far back in time. Optimising the SE(2) graph gives corrected poses and a merged cone map.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from fs_slam.ekf_slam import relative_pose, wrap


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pose composition a (+) b: pose b given in frame a, expressed in a's parent frame."""
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])])


def transform_pts(pose: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Transform (N,2) points from the frame of `pose` into its parent frame."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return pts @ np.array([[c, s], [-s, c]]) + pose[:2]


@dataclass
class Keyframe:
    odom: np.ndarray            # raw odometry pose (x, y, yaw); never modified
    pose: np.ndarray            # current estimate, initialised from odom, updated by optimise()
    xy: np.ndarray              # (N,2) cones in the car frame
    color: np.ndarray           # (N,) cone colours


@dataclass
class GraphParams:
    kf_dist: float = 1.5        # new keyframe after this much travel [m]...
    kf_yaw: float = 0.35        # ...or this much heading change [rad]
    loop_min_gap: int = 20      # candidate must be >= N keyframes older ("re-observed after N steps")
    loop_radius: float = 8.0    # candidate must be within this distance (estimated) [m]
    icp_iters: int = 15
    icp_gate: float = 0.8       # final max correspondence distance in ICP [m] (starts at 2x)
    min_inliers: int = 5
    max_rms: float = 0.25       # accept loop if inlier RMS below this [m]
    max_shift: float = 1.4      # reject loops that move the odometry guess by more than this [m]
    max_turn: float = 0.15      # ...or this [rad]; periodic cone rows fool ICP into matches
                                # that are one cone spacing off
    merge_eps: float = 0.5      # cones closer than this (after correction) are one cone [m]
    min_obs: int = 2            # keyframes that must see a cone for it to enter the map


def icp(src: np.ndarray, src_col: np.ndarray, dst: np.ndarray, dst_col: np.ndarray,
        T0: np.ndarray, p: GraphParams) -> Tuple[np.ndarray, int, float]:
    """Align src cones to dst cones. T maps src frame -> dst frame (x, y, yaw).

    Returns (T, inlier count, inlier RMS). Colours must agree unless one is unknown (0).
    """
    T, tree = T0.copy(), cKDTree(dst)
    n_in, rms = 0, np.inf
    for it in range(p.icp_iters):
        gate = p.icp_gate * (1 + 1.5 * (1 - it / (p.icp_iters - 1)))   # coarse-to-fine: 2.0x -> 1.0x
        moved = transform_pts(T, src)
        d, k = tree.query(moved)
        ok = d < gate
        ok &= (src_col == 0) | (dst_col[k] == 0) | (src_col == dst_col[k])
        if ok.sum() < 3:
            return T, int(ok.sum()), np.inf
        a, b = moved[ok], dst[k[ok]]
        ca, cb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))          # Kabsch rigid alignment
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        dth = np.arctan2(R[1, 0], R[0, 0])
        dT = np.array([*(cb - R @ ca), dth])                     # correction in dst frame
        T = np.array([*(R @ T[:2] + dT[:2]), wrap(T[2] + dth)])
        n_in, rms = int(ok.sum()), float(np.sqrt(np.mean(d[ok] ** 2)))   # (re-evaluated next pass)
    return T, n_in, rms


class PoseGraph:
    def __init__(self, p: GraphParams = GraphParams()) -> None:
        self.p = p
        self.kfs: List[Keyframe] = []
        self.edges: List[Tuple[int, int, np.ndarray, np.ndarray]] = []   # (i, j, z_ij, sigmas)
        self.n_loops = 0

    def needs_keyframe(self, pose: np.ndarray) -> bool:
        """True when the car has moved/turned enough since the last keyframe."""
        if not self.kfs:
            return True
        rel = relative_pose(self.kfs[-1].odom, pose)
        return bool(np.hypot(rel[0], rel[1]) > self.p.kf_dist or abs(rel[2]) > self.p.kf_yaw)

    def add_keyframe(self, pose: np.ndarray, xy: np.ndarray, color: np.ndarray) -> bool:
        """Add a keyframe + odometry edge; try a loop closure. Returns True if a loop closed."""
        k = len(self.kfs)
        self.kfs.append(Keyframe(pose.copy(), pose.copy(), xy.copy(), color.copy()))
        if k > 0:
            z = relative_pose(self.kfs[k - 1].odom, pose)   # odometry edge from raw odometry
            d = np.hypot(z[0], z[1])
            self.edges.append((k - 1, k, z, np.array([0.05 * d + 0.02, 0.05 * d + 0.02, 0.02 + 0.05 * abs(z[2])])))
        loop = self._try_loop(k)
        if loop:
            self.optimise()
        return loop

    def _try_loop(self, j: int) -> bool:
        """Look for the best old keyframe that aligns with keyframe j; add a loop edge."""
        kj = self.kfs[j]
        if len(kj.xy) < self.p.min_inliers:
            return False
        best: Optional[Tuple[int, np.ndarray, int, float]] = None
        for i in range(0, j - self.p.loop_min_gap + 1):
            ki = self.kfs[i]
            if len(ki.xy) < self.p.min_inliers or np.linalg.norm(ki.pose[:2] - kj.pose[:2]) > self.p.loop_radius:
                continue
            T0 = relative_pose(ki.pose, kj.pose)               # current estimate of the relative pose
            T, n_in, rms = icp(kj.xy, kj.color, ki.xy, ki.color, T0, self.p)
            if np.hypot(*(T[:2] - T0[:2])) > self.p.max_shift or abs(wrap(T[2] - T0[2])) > self.p.max_turn:
                continue                                       # implausible correction: likely a false match
            if n_in >= self.p.min_inliers and rms < self.p.max_rms and (best is None or n_in > best[2]):
                best = (i, T, n_in, rms)
        if best is None:
            return False
        i, T, _, _ = best
        self.edges.append((i, j, T, np.array([0.1, 0.1, 0.03])))   # trusted more than odometry
        self.n_loops += 1
        return True

    def optimise(self) -> None:
        """Least-squares SE(2) pose-graph optimisation. Keyframe 0 is held fixed (the anchor)."""
        K, E = len(self.kfs), len(self.edges)
        X0 = np.array([kf.pose for kf in self.kfs])
        I = np.array([e[0] for e in self.edges])               # edge arrays, built once per solve
        J = np.array([e[1] for e in self.edges])
        Z = np.array([e[2] for e in self.edges])
        S = np.array([e[3] for e in self.edges])
        def residual(x: np.ndarray) -> np.ndarray:
            X = np.vstack([X0[0], x.reshape(-1, 3)])
            d = X[J, :2] - X[I, :2]                              # vectorised relative_pose(Xi, Xj)
            c, s = np.cos(X[I, 2]), np.sin(X[I, 2])
            rel = np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], X[J, 2] - X[I, 2]])
            e = rel - Z
            e[:, 2] = wrap(e[:, 2])
            return (e / S).ravel()                               # whitened by edge uncertainty

        sol = least_squares(residual, X0[1:].ravel(), max_nfev=30)
        for kf, pose in zip(self.kfs[1:], sol.x.reshape(-1, 3)):
            kf.pose = np.array([pose[0], pose[1], wrap(pose[2])])

    def cone_map(self) -> Tuple[np.ndarray, np.ndarray]:
        """Merge all keyframe cones (using current poses) into a map: (positions, colours)."""
        pts = [transform_pts(kf.pose, kf.xy) for kf in self.kfs if len(kf.xy)]
        if not pts:
            return np.zeros((0, 2)), np.zeros(0, dtype=int)
        P, C = np.vstack(pts), np.concatenate([kf.color for kf in self.kfs if len(kf.xy)])
        pairs = cKDTree(P).query_pairs(self.p.merge_eps, output_type='ndarray')
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P),) * 2)
        _, lab = connected_components(g, directed=False)
        out, cols = [], []
        for l in np.unique(lab):
            m = lab == l
            if m.sum() < self.p.min_obs:
                continue
            known = C[m][C[m] > 0]
            out.append(P[m].mean(0))
            cols.append(int(np.bincount(known).argmax()) if len(known) else 0)   # majority colour
        return np.array(out).reshape(-1, 2), np.array(cols, dtype=int)
