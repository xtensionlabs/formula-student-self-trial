"""Path + speed profile as a lookup (ROS-free): MPC reference horizon and ANN features."""
from typing import Optional, Tuple
import numpy as np


def wrap(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class PathRef:
    """Centerline samples with arc length, heading, curvature and target speed."""

    def __init__(self, xy: np.ndarray, v: np.ndarray, v_end: float = 2.0, a_brake: float = 8.0,
                 closed: Optional[bool] = None) -> None:
        self.xy, self.v_prof = np.asarray(xy, float), np.asarray(v, float)
        d = np.diff(self.xy, axis=0)
        self.s = np.r_[0.0, np.cumsum(np.hypot(d[:, 0], d[:, 1]))]
        self.psi = np.unwrap(np.arctan2(np.gradient(self.xy[:, 1]), np.gradient(self.xy[:, 0])))
        dpsi = np.gradient(self.psi, self.s)                       # curvature = d psi / d s (signed)
        self.kappa = dpsi
        # Closed loop = last sample meets the first (the planner publishes one loop starting at the car)
        self.closed = bool(closed) if closed is not None else bool(
            len(xy) > 40 and np.hypot(*(self.xy[-1] - self.xy[0])) < 3.0)
        self.L = self.s[-1] + (np.hypot(*(self.xy[0] - self.xy[-1])) if self.closed else 0.0)
        self.v_end, self.a_brake = v_end, a_brake
        self.last = 0                                              # last nearest index (search window)

    def nearest(self, pos: np.ndarray, window: int = 80) -> int:
        """Index of the closest path sample, searched near the previous result (fast, no jumps)."""
        n = len(self.xy)
        idx = (np.arange(self.last - window // 4, self.last + window) % n) if self.closed else \
            np.arange(max(0, self.last - window // 4), min(n, self.last + window))
        if len(idx) == 0:
            idx = np.arange(n)
        self.last = int(idx[np.argmin(np.linalg.norm(self.xy[idx] - pos, axis=1))])
        return self.last

    def at(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate (xy, psi, kappa, v_ref) at arc lengths s (wraps on closed paths)."""
        s = np.asarray(s, float)
        if self.closed:
            s = s % self.L
            sx = np.r_[self.s, self.L]
            xy = np.vstack([self.xy, self.xy[:1]])
            psi = np.r_[self.psi, self.psi[0] + 2 * np.pi * np.round((self.psi[-1] - self.psi[0]) / (2 * np.pi))]
            kap, v = np.r_[self.kappa, self.kappa[0]], np.r_[self.v_prof, self.v_prof[0]]
        else:
            sx, xy, psi, kap, v = self.s, self.xy, self.psi, self.kappa, self.v_prof
        out_xy = np.column_stack([np.interp(s, sx, xy[:, 0]), np.interp(s, sx, xy[:, 1])])
        v_ref = np.interp(s, sx, v)
        if not self.closed:                                        # slow down toward the end of an open path
            v_ref = np.minimum(v_ref, np.sqrt(self.v_end ** 2 + 2 * self.a_brake * np.maximum(self.L - s, 0)))
        return out_xy, np.interp(s, sx, psi), np.interp(s, sx, kap), v_ref

    def horizon(self, pos: np.ndarray, v: float, N: int, dt: float, v_floor: float = 2.0) -> np.ndarray:
        """Reference (4, N+1) = [x, y, psi, v_ref] for the MPC, spaced by the car's current speed."""
        s0 = self.s[self.nearest(pos)]
        s = s0 + np.arange(N + 1) * dt * max(v, v_floor)
        xy, psi, _, v_ref = self.at(s)
        return np.vstack([xy.T, psi, v_ref])

    def features(self, pose: np.ndarray, v: float, preview: float = 0.25) -> np.ndarray:
        """ANN input [cte, heading_err, speed_err, curvature]; cte > 0 = car left of the path."""
        i = self.nearest(pose[:2])
        xy, psi, _, v_ref = self.at(np.array([self.s[i]]))
        cte = -np.sin(psi[0]) * (pose[0] - xy[0, 0]) + np.cos(psi[0]) * (pose[1] - xy[0, 1])
        _, _, kap, _ = self.at(np.array([self.s[i] + max(v, 2.0) * preview]))   # curvature a bit ahead
        return np.array([cte, wrap(pose[2] - psi[0]), v - v_ref[0], kap[0]])
