"""EKF-SLAM with cones as landmarks (ROS-free).

State x = [px, py, yaw | l1x, l1y | l2x, l2y | ...] in the map frame (origin = start pose).
Motion: odometry delta (expressed in the previous car frame). Observation: range + bearing.
Colours (0 unknown, 1 yellow, 2 blue, 3 orange) are tracked per landmark by confidence-weighted
votes; association only compares a cone with landmarks of a compatible colour, so yellow and
blue cones form separate maps inside one filter.
"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


def wrap(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def relative_pose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pose b expressed in the frame of pose a. Poses are (x, y, yaw)."""
    c, s = np.cos(a[2]), np.sin(a[2])
    d = b[:2] - a[:2]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], wrap(b[2] - a[2])])


@dataclass
class SlamParams:
    sig_xy_per_m: float = 0.05     # odometry position noise per metre driven
    sig_xy_floor: float = 0.005
    sig_yaw_per_rad: float = 0.05  # odometry heading noise per radian turned
    sig_yaw_floor: float = 0.002
    min_range_sigma: float = 0.1   # floor on measurement range std [m]
    min_bearing_sigma: float = 0.02  # floor on bearing std [rad]
    match_gate: float = 9.21       # chi2(2 dof, 99%): accept association below this
    new_gate: float = 25.0         # start a new landmark only above this (else: ambiguous, skip)
    search_radius: float = 3.0     # only test landmarks this close to the observation [m]
    confirm_hits: int = 3          # observations before a landmark is published


class EkfSlam:
    def __init__(self, p: SlamParams = SlamParams()) -> None:
        self.p = p
        self.x = np.zeros(3)
        self.P = np.zeros((3, 3))
        self.hits: List[int] = []                  # observation count per landmark
        self.votes: List[np.ndarray] = []          # per-landmark colour votes (4,)

    @property
    def n(self) -> int:
        return len(self.hits)

    def predict(self, d: np.ndarray) -> None:
        """Apply odometry delta d = (dx, dy, dyaw) given in the previous car frame."""
        c, s = np.cos(self.x[2]), np.sin(self.x[2])
        self.x[0] += c * d[0] - s * d[1]
        self.x[1] += s * d[0] + c * d[1]
        self.x[2] = wrap(self.x[2] + d[2])
        # Jacobian of the pose w.r.t. itself, and w.r.t. the delta (noise injection)
        F = np.array([[1, 0, -s * d[0] - c * d[1]], [0, 1, c * d[0] - s * d[1]], [0, 0, 1]])
        G = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        dist = np.hypot(d[0], d[1])
        q = np.diag([(self.p.sig_xy_per_m * dist + self.p.sig_xy_floor) ** 2] * 2
                    + [(self.p.sig_yaw_per_rad * abs(d[2]) + self.p.sig_yaw_floor) ** 2])
        self.P[:3, :] = F @ self.P[:3, :]            # only the pose rows/cols change
        self.P[:, :3] = self.P[:, :3] @ F.T
        self.P[:3, :3] += G @ q @ G.T

    def _rb_cov(self, xy: np.ndarray, cov: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """Range, bearing and their covariance from a vehicle-frame cone position + xy covariance."""
        r = max(np.hypot(*xy), 1e-3)
        J = np.array([[xy[0] / r, xy[1] / r], [-xy[1] / r ** 2, xy[0] / r ** 2]])
        R = J @ cov @ J.T
        R[0, 0] = max(R[0, 0], self.p.min_range_sigma ** 2)
        R[1, 1] = max(R[1, 1], self.p.min_bearing_sigma ** 2)
        return r, np.arctan2(xy[1], xy[0]), R

    def _jac(self, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Observation Jacobians (pose part, landmark part) and predicted (range, bearing)
        for landmark offset (dx, dy) from the car."""
        q = dx * dx + dy * dy
        r = np.sqrt(q)
        Hp = np.array([[-dx / r, -dy / r, 0], [dy / q, -dx / q, -1]])
        Hl = np.array([[dx / r, dy / r], [-dy / q, dx / q]])
        return Hp, Hl, r, wrap(np.arctan2(dy, dx) - self.x[2])

    def update(self, obs: np.ndarray, colors: np.ndarray, conf: np.ndarray,
               covs: np.ndarray) -> None:
        """Process one scan. obs (M,2) vehicle-frame xy, colors (M,), conf (M,), covs (M,2,2)."""
        for xy, col, cf, cov in zip(obs, colors, conf, covs):
            r, b, R = self._rb_cov(xy, cov)
            j = self._associate(r, b, R, int(col))
            if j == -2:
                continue                                 # ambiguous: skip this observation
            if j == -1:
                self._add_landmark(r, b, R)
                j = self.n - 1
            self._correct(j, r, b, R)
            self.hits[j] += 1
            if col:
                self.votes[j][col] += cf                 # colour vote weighted by confidence

    def _associate(self, r: float, b: float, R: np.ndarray, col: int) -> int:
        """Nearest-neighbour association with Mahalanobis gating.

        Returns landmark index, -1 for 'new landmark', -2 for 'ambiguous, ignore'.
        """
        if self.n == 0:
            return -1
        L = self.x[3:].reshape(-1, 2)
        pred = self.x[:2] + r * np.array([np.cos(b + self.x[2]), np.sin(b + self.x[2])])
        near = np.where(np.linalg.norm(L - pred, axis=1) < self.p.search_radius)[0]
        best, best_d = -1, np.inf
        for j in near:
            vote = self.votes[j]
            lm_col = int(np.argmax(vote[1:]) + 1) if vote[1:].any() else 0
            if col and lm_col and col != lm_col:
                continue                                 # colours disagree: cannot be the same cone
            dx, dy = L[j] - self.x[:2]
            Hp, Hl, rh, bh = self._jac(dx, dy)
            i = 3 + 2 * j
            Pp, Pl, Ppl = self.P[:3, :3], self.P[i:i + 2, i:i + 2], self.P[:3, i:i + 2]
            S = Hp @ Pp @ Hp.T + Hl @ Pl @ Hl.T + Hp @ Ppl @ Hl.T + Hl @ Ppl.T @ Hp.T + R
            v = np.array([r - rh, wrap(b - bh)])
            d2 = v @ np.linalg.solve(S, v)               # squared Mahalanobis distance
            if d2 < best_d:
                best, best_d = j, d2
        if best_d < self.p.match_gate:
            return best
        return -1 if best_d > self.p.new_gate else -2

    def _add_landmark(self, r: float, b: float, R: np.ndarray) -> None:
        """Augment state and covariance with a new landmark seen at (r, b)."""
        yaw = self.x[2]
        a = yaw + b
        lm = self.x[:2] + r * np.array([np.cos(a), np.sin(a)])
        Gp = np.array([[1, 0, -r * np.sin(a)], [0, 1, r * np.cos(a)]])        # d lm / d pose
        Gz = np.array([[np.cos(a), -r * np.sin(a)], [np.sin(a), r * np.cos(a)]])  # d lm / d (r,b)
        n = len(self.x)
        P = np.zeros((n + 2, n + 2))
        P[:n, :n] = self.P
        cross = Gp @ self.P[:3, :]                       # landmark-vs-everything covariance
        P[n:, :n], P[:n, n:] = cross, cross.T
        P[n:, n:] = Gp @ self.P[:3, :3] @ Gp.T + Gz @ R @ Gz.T
        self.P, self.x = P, np.concatenate([self.x, lm])
        self.hits.append(0)
        self.votes.append(np.zeros(4))

    def _correct(self, j: int, r: float, b: float, R: np.ndarray) -> None:
        """EKF correction with landmark j (sparse Jacobian: pose + that landmark only)."""
        i = 3 + 2 * j
        dx, dy = self.x[i:i + 2] - self.x[:2]
        Hp, Hl, rh, bh = self._jac(dx, dy)
        cols = np.r_[0:3, i:i + 2]
        H = np.hstack([Hp, Hl])
        PHt = self.P[:, cols] @ H.T                      # (n,2)
        S = H @ PHt[cols] + R
        K = PHt @ np.linalg.inv(S)
        self.x += K @ np.array([r - rh, wrap(b - bh)])
        self.x[2] = wrap(self.x[2])
        self.P -= K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def landmarks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Confirmed landmarks: (positions (K,2), colours (K,))."""
        ok = [j for j in range(self.n) if self.hits[j] >= self.p.confirm_hits]
        pos = self.x[3:].reshape(-1, 2)[ok]
        col = np.array([int(np.argmax(self.votes[j][1:]) + 1) if self.votes[j][1:].any() else 0
                        for j in ok], dtype=int)
        return pos, col
