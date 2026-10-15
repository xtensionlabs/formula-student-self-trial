"""Unscented Kalman Filter (Merwe sigma points), ROS-free so it is unit-testable.

State: [x, y, yaw, vx, vy, yaw_rate]  (x,y,yaw in odom frame; velocities in body frame).
Process model: kinematic bicycle (see `process`).
"""
from typing import Callable, Sequence
import numpy as np

N = 6  # state dimension


def wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def process(x: np.ndarray, ax: float, dt: float, lr: float = 0.765) -> np.ndarray:
    """Kinematic bicycle process model.

    Pose is integrated from body-frame velocities. Longitudinal speed follows the
    IMU acceleration `ax`. Lateral speed is set to the no-slip bicycle value
    vy = lr * yaw_rate (lr = CoG-to-rear-axle distance).
    """
    px, py, yaw, vx, vy, r = x
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        px + (vx * c - vy * s) * dt,   # world-frame position update
        py + (vx * s + vy * c) * dt,
        wrap(yaw + r * dt),            # heading update
        vx + ax * dt,                  # speed driven by measured acceleration
        lr * r,                        # kinematic (no-slip) lateral velocity
        r,                             # yaw rate: random walk
    ])


def _resid(a: np.ndarray, b: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    """a - b with the listed columns wrapped as angles."""
    r = a - b
    for i in idx:
        r[..., i] = (r[..., i] + np.pi) % (2 * np.pi) - np.pi
    return r


class UKF:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray,
                 alpha: float = 0.5, beta: float = 2.0, kappa: float = 0.0) -> None:
        self.x, self.P, self.Q = x0.astype(float), P0.astype(float), Q.astype(float)
        self.lam = alpha ** 2 * (N + kappa) - N
        # Sigma-point weights for mean (wm) and covariance (wc)
        self.wm = np.full(2 * N + 1, 0.5 / (N + self.lam))
        self.wc = self.wm.copy()
        self.wm[0] = self.lam / (N + self.lam)
        self.wc[0] = self.wm[0] + (1 - alpha ** 2 + beta)

    def _sigma(self) -> np.ndarray:
        """2N+1 sigma points around the current mean (one per row)."""
        L = np.linalg.cholesky((N + self.lam) * self.P)
        return np.array([self.x] + [self.x + L[:, i] for i in range(N)]
                        + [self.x - L[:, i] for i in range(N)])

    def _mean(self, pts: np.ndarray, angle_idx: Sequence[int]) -> np.ndarray:
        """Weighted mean; angle columns are averaged on the unit circle."""
        m = self.wm @ pts
        for i in angle_idx:
            m[i] = np.arctan2(self.wm @ np.sin(pts[:, i]), self.wm @ np.cos(pts[:, i]))
        return m

    def predict(self, ax: float, dt: float) -> None:
        """Propagate sigma points through the process model, add process noise Q*dt."""
        pts = np.array([process(p, ax, dt) for p in self._sigma()])
        self.x = self._mean(pts, [2])
        d = _resid(pts, self.x, [2])
        self.P = (self.wc[:, None] * d).T @ d + self.Q * dt

    def update(self, z: np.ndarray, h: Callable[[np.ndarray], np.ndarray], R: np.ndarray,
               z_angle_idx: Sequence[int] = ()) -> None:
        """Generic measurement update. `z_angle_idx`: entries of z / h(x) that are angles."""
        pts = self._sigma()
        Z = np.array([h(p) for p in pts])
        zm = self._mean(Z, z_angle_idx)
        dz, dx = _resid(Z, zm, z_angle_idx), _resid(pts, self.x, [2])
        S = (self.wc[:, None] * dz).T @ dz + R           # innovation covariance
        Pxz = (self.wc[:, None] * dx).T @ dz             # state/measurement cross covariance
        K = Pxz @ np.linalg.inv(S)                       # Kalman gain
        self.x = self.x + K @ _resid(z, zm, z_angle_idx)
        self.x[2] = wrap(self.x[2])
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)               # keep symmetric


# --- measurement models used by the node ------------------------------------------------
def h_imu(x: np.ndarray) -> np.ndarray:
    """IMU observes heading and yaw rate."""
    return np.array([x[2], x[5]])


def h_gss_body(x: np.ndarray) -> np.ndarray:
    """GSS reported in the body frame."""
    return x[3:5]


def h_gss_world(x: np.ndarray) -> np.ndarray:
    """GSS reported in the world frame: rotate body velocity by yaw."""
    c, s = np.cos(x[2]), np.sin(x[2])
    return np.array([x[3] * c - x[4] * s, x[3] * s + x[4] * c])


def make_h_bicycle(lr: float = 0.765) -> Callable[[np.ndarray], np.ndarray]:
    """Pseudo-measurement (z = 0) enforcing vy - lr*r = 0, the kinematic bicycle constraint."""
    return lambda x: np.array([x[4] - lr * x[5]])
