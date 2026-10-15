"""Speed profile: Gaussian-process curvature -> safe speed, plus accel/brake limited profile."""
from typing import Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


class SpeedGP:
    """GP regression of max safe speed vs curvature, prior = physics v = sqrt(a_lat / kappa)."""

    def __init__(self, a_lat: float = 15.0, v_cap: float = 15.0, max_points: int = 400) -> None:
        self.a_lat, self.v_cap, self.max_points = a_lat, v_cap, max_points
        k = np.linspace(0.0, 0.5, 120)                    # curvature 0 .. 0.5 1/m (radius >= 2 m)
        self.X = k[:, None]
        self.y = np.minimum(v_cap, np.sqrt(a_lat / np.maximum(k, 1e-6)))   # physics training data
        self.gp = GaussianProcessRegressor(
            ConstantKernel(25.0) * RBF(0.1) + WhiteKernel(0.05), normalize_y=True, n_restarts_optimizer=0)
        self.gp.fit(self.X, self.y)

    def predict(self, kappa: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(mean speed, std) for each curvature, clipped to [2, v_cap]."""
        m, s = self.gp.predict(np.asarray(kappa)[:, None], return_std=True)
        return np.clip(m, 2.0, self.v_cap), s

    def update(self, kappa: np.ndarray, v_safe: np.ndarray) -> None:
        """Online refinement: add observed (curvature, safe speed) pairs and refit the posterior."""
        self.X = np.vstack([self.X, np.asarray(kappa)[:, None]])[-self.max_points:]
        self.y = np.concatenate([self.y, v_safe])[-self.max_points:]
        self.gp.fit(self.X, self.y)


def limit_accel(s: np.ndarray, v: np.ndarray, a_acc: float, a_brake: float, closed: bool) -> np.ndarray:
    """Forward/backward pass so the profile respects acceleration and braking limits."""
    v = v.copy()
    ds = np.diff(s, append=s[-1] + (s[1] - s[0] if len(s) > 1 else 1.0))
    reps = 2 if closed else 1                             # closed track: run twice to wrap around
    for _ in range(reps):
        for i in range(len(v) - 2, -1, -1):               # backward: brake in time for slow corners
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_brake * ds[i]))
        if closed:
            v[-1] = min(v[-1], np.sqrt(v[0] ** 2 + 2 * a_brake * ds[-1]))
    for _ in range(reps):
        for i in range(1, len(v)):                        # forward: limited acceleration
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * a_acc * ds[i - 1]))
        if closed:
            v[0] = min(v[0], np.sqrt(v[-1] ** 2 + 2 * a_acc * ds[-1]))
    return v
