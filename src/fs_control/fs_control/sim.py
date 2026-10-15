"""Closed-loop simulator for offline controller tuning / training / tests (ROS-free).

Plant = kinematic bicycle with actuator lag, a steering rate limit, wheelbase / acceleration
mismatch and pose noise, so controllers tuned here are not tuned on their own model.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import numpy as np
from fs_control.reference import PathRef, wrap


def random_track(rng: np.random.Generator, length: float = 200.0, ds: float = 0.5,
                 v_max: float = 12.0, a_lat: float = 8.0) -> PathRef:
    """Open track made of random constant-curvature segments (radius 6..40 m), smoothed."""
    n = int(length / ds)
    kappa, i = np.zeros(n), 0
    while i < n:
        seg = int(rng.uniform(8, 35) / ds)
        r = rng.uniform(6, 40) * rng.choice([-1, 1]) if rng.random() < 0.7 else np.inf
        kappa[i:i + seg] = 0.0 if np.isinf(r) else 1 / r
        i += seg
    kappa = np.convolve(kappa, np.ones(9) / 9, mode='same')            # clothoid-like transitions
    psi = np.cumsum(kappa) * ds
    xy = np.cumsum(np.column_stack([np.cos(psi), np.sin(psi)]), axis=0) * ds
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-3)))
    for _ in range(3):                                                 # smooth braking ramps
        v = np.minimum(v, np.sqrt(np.r_[v[1:], v[-1]] ** 2 + 2 * 6.0 * ds))
    return PathRef(xy, v, closed=False)


@dataclass
class PlantParams:
    L: float = 1.53 * 1.04          # true wheelbase differs from the controller's model
    lr: float = 0.765
    tau_steer: float = 0.08         # steering actuator lag [s]
    steer_rate: float = 1.2         # rad/s limit
    tau_acc: float = 0.15           # drive-train lag [s]
    acc_gain: float = 0.9           # actual accel = gain * commanded
    pose_noise: float = 0.03        # [m] noise on the measured pose
    d_max: float = 0.436


class Plant:
    def __init__(self, x0: np.ndarray, p: PlantParams = PlantParams(), seed: int = 0) -> None:
        self.x, self.p, self.rng = x0.astype(float), p, np.random.default_rng(seed)
        self.delta, self.acc = 0.0, 0.0

    def step(self, u: np.ndarray, dt: float, sub: int = 5) -> None:
        """Advance dt seconds with command u = [delta, a] (applied through actuator lag)."""
        p = self.p
        for _ in range(sub):
            h = dt / sub
            dd = np.clip((np.clip(u[0], -p.d_max, p.d_max) - self.delta) / p.tau_steer, -p.steer_rate, p.steer_rate)
            self.delta += dd * h
            self.acc += (p.acc_gain * u[1] - self.acc) / p.tau_acc * h
            X, Y, psi, v = self.x
            beta = np.arctan(p.lr / p.L * np.tan(self.delta))
            self.x = np.array([X + v * np.cos(psi + beta) * h, Y + v * np.sin(psi + beta) * h,
                               psi + v / p.lr * np.sin(beta) * h, max(v + self.acc * h, 0.0)])

    def measure(self) -> np.ndarray:
        """Noisy measured state."""
        return self.x + self.rng.normal(0, [self.p.pose_noise] * 2 + [0.002, 0.02], 4)


Controller = Callable[[np.ndarray, PathRef, np.ndarray], Optional[np.ndarray]]


def run(controller: Controller, ref: PathRef, T: float = 20.0, dt: float = 0.05, seed: int = 0,
        offset: float = 0.3, v0: float = 3.0) -> Dict[str, np.ndarray]:
    """Drive `controller(x_meas, ref, u_prev) -> [delta, a]` along `ref`; return logged signals.

    The car starts `offset` m beside the path. The run stops early if |cte| > 2.5 m (off track).
    """
    ref.last = 0
    pose0 = np.array([*ref.xy[0], ref.psi[0], v0])
    pose0[:2] += offset * np.array([-np.sin(pose0[2]), np.cos(pose0[2])])
    plant, u, log = Plant(pose0, seed=seed), np.zeros(2), []
    for k in range(int(T / dt)):
        xm = plant.measure()
        feat = ref.features(xm[:3], xm[3])
        u_new = controller(xm, ref, u)
        u = u if u_new is None else u_new
        log.append([*feat, *u, plant.x[3], ref.s[ref.last]])
        plant.step(u, dt)
        if abs(feat[0]) > 2.5 or ref.s[ref.last] > ref.s[-1] - 3:
            break
    a = np.array(log)
    return {'feat': a[:, :4], 'u': a[:, 4:6], 'v': a[:, 6], 's': a[:, 7], 'dt': dt,
            'off_track': bool(np.abs(a[:, 0]).max() > 2.5)}


def tracking_cost(r: Dict[str, np.ndarray]) -> float:
    """Scalar score of a run (lower is better): tracking error + smoothness + progress."""
    f, u, dt = r['feat'], r['u'], r['dt']
    steer_rate = np.diff(u[:, 0]) / dt if len(u) > 1 else np.zeros(1)
    c = (np.sqrt(np.mean(f[:, 0] ** 2)) + 0.5 * np.sqrt(np.mean(f[:, 1] ** 2))
         + 0.05 * np.sqrt(np.mean(f[:, 2] ** 2)) + 0.05 * np.sqrt(np.mean(steer_rate ** 2)))
    c -= 0.01 * r['s'][-1]                                           # reward distance covered
    return float(c + (50.0 if r['off_track'] else 0.0))


def mpc_controller(mpc, w, record: Optional[list] = None, noise: Tuple[float, float] = (0.0, 0.0),
                   seed: int = 0) -> Controller:
    """Wrap an `Mpc` as a sim controller. `record` collects (features, mpc_command) training pairs;
    `noise` (sigma_delta, sigma_a) perturbs the *executed* command (exploration for imitation data)
    while the recorded label stays the clean MPC command."""
    rng = np.random.default_rng(seed)

    def ctrl(x: np.ndarray, ref: PathRef, u_prev: np.ndarray) -> Optional[np.ndarray]:
        u, _ = mpc.solve(x, ref.horizon(x[:2], x[3], mpc.N, mpc.dt), u_prev, w)
        if u is None:
            return None
        if record is not None:
            record.append(np.r_[ref.features(x[:3], x[3]), u])
        return u + rng.normal(0, noise)
    return ctrl
