"""Nonlinear MPC on a kinematic bicycle (CasADi + IPOPT, ROS-free).

State x = [X, Y, psi, v], input u = [delta (steering angle), a (acceleration)].
The same `dynamics` and `stage_cost` expressions are used inside the optimiser and (called with
numbers) in tests / the CMA-ES tuner, so there is a single definition of model and cost.
"""
import time
from dataclasses import dataclass, fields
from typing import Dict, Optional, Tuple
import casadi as ca
import numpy as np


@dataclass
class MpcWeights:
    q_ct: float = 4.0       # cross-track error^2
    q_head: float = 3.0     # heading error  (2*(1-cos))
    q_v: float = 0.5        # speed error^2
    r_d: float = 0.5        # steering magnitude^2
    r_a: float = 0.02       # acceleration magnitude^2
    r_dd: float = 8.0       # steering change^2 between steps (smoothness)
    r_da: float = 0.05      # acceleration change^2

    def vec(self) -> np.ndarray:
        return np.array([getattr(self, f.name) for f in fields(self)])


@dataclass
class VehicleLimits:
    L: float = 1.53         # wheelbase [m]
    lr: float = 0.765       # CoG -> rear axle [m]
    d_max: float = 0.436    # steering limit [rad] (FSDS: 25 deg)
    a_min: float = -10.0    # braking limit [m/s^2]
    a_max: float = 5.0      # traction limit [m/s^2]


def make_dynamics(lim: VehicleLimits, dt: float) -> ca.Function:
    """One RK4 step of the kinematic bicycle (slip angle at the CoG)."""
    x, u = ca.SX.sym('x', 4), ca.SX.sym('u', 2)

    def f(x, u):
        beta = ca.atan(lim.lr / lim.L * ca.tan(u[0]))            # slip angle at the CoG
        return ca.vertcat(x[3] * ca.cos(x[2] + beta), x[3] * ca.sin(x[2] + beta),
                          x[3] / lim.lr * ca.sin(beta), u[1])

    k1 = f(x, u); k2 = f(x + dt / 2 * k1, u); k3 = f(x + dt / 2 * k2, u); k4 = f(x + dt * k3, u)
    return ca.Function('step', [x, u], [x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)])


def make_stage_cost() -> ca.Function:
    """Cost of one stage: state error vs reference [xr, yr, psir, vr] + input terms."""
    x, r, u, up, w = (ca.SX.sym(n, d) for n, d in (('x', 4), ('r', 4), ('u', 2), ('up', 2), ('w', 7)))
    cte = -ca.sin(r[2]) * (x[0] - r[0]) + ca.cos(r[2]) * (x[1] - r[1])   # signed lateral error
    head = 2 * (1 - ca.cos(x[2] - r[2]))                                  # smooth, wrap-free
    J = (w[0] * cte ** 2 + w[1] * head + w[2] * (x[3] - r[3]) ** 2
         + w[3] * u[0] ** 2 + w[4] * u[1] ** 2
         + w[5] * (u[0] - up[0]) ** 2 + w[6] * (u[1] - up[1]) ** 2)
    return ca.Function('stage_cost', [x, r, u, up, w], [J])


class Mpc:
    def __init__(self, N: int = 20, dt: float = 0.05, lim: VehicleLimits = VehicleLimits(),
                 time_limit: float = 0.045) -> None:
        self.N, self.dt, self.lim = N, dt, lim
        self.step, self.cost = make_dynamics(lim, dt), make_stage_cost()
        o = self.opti = ca.Opti()
        self.X, self.U = o.variable(4, N + 1), o.variable(2, N)
        self.p_x0, self.p_ref = o.parameter(4), o.parameter(4, N + 1)
        self.p_up, self.p_w = o.parameter(2), o.parameter(7)
        J = 0
        for k in range(N):
            up = self.p_up if k == 0 else self.U[:, k - 1]            # previous input for smoothness
            J += self.cost(self.X[:, k + 1], self.p_ref[:, k + 1], self.U[:, k], up, self.p_w)
            o.subject_to(self.X[:, k + 1] == self.step(self.X[:, k], self.U[:, k]))   # dynamics
        o.minimize(J)
        o.subject_to(self.X[:, 0] == self.p_x0)
        o.subject_to(o.bounded(-lim.d_max, self.U[0, :], lim.d_max))
        o.subject_to(o.bounded(lim.a_min, self.U[1, :], lim.a_max))
        o.subject_to(self.X[3, :] >= 0)                                # no reversing
        o.solver('ipopt', {'print_time': False, 'ipopt.print_level': 0, 'ipopt.sb': 'yes',
                           'ipopt.max_cpu_time': time_limit, 'ipopt.max_iter': 50,
                           'ipopt.tol': 1e-3, 'ipopt.acceptable_tol': 1e-2,
                           'ipopt.warm_start_init_point': 'yes', 'ipopt.mu_strategy': 'adaptive'})
        self.Xg = np.zeros((4, N + 1)); self.Ug = np.zeros((2, N))   # warm-start guess

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """One model step with numbers (e.g. to compensate computation delay)."""
        return np.array(self.step(x, u)).ravel()

    def solve(self, x0: np.ndarray, ref: np.ndarray, u_prev: np.ndarray,
              w: MpcWeights) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        """Return (first input [delta, a] or None on failure, info with solve time in ms)."""
        o = self.opti
        o.set_value(self.p_x0, x0); o.set_value(self.p_ref, ref)
        o.set_value(self.p_up, u_prev); o.set_value(self.p_w, w.vec())
        if not np.any(self.Xg):                                        # first solve: start on the reference
            self.Xg = ref.copy(); self.Xg[:, 0] = x0
        o.set_initial(self.X, self.Xg); o.set_initial(self.U, self.Ug)
        t0 = time.perf_counter()
        try:
            sol = o.solve()
            ok = True
        except RuntimeError:                                           # infeasible / time limit / max_iter
            sol, ok = None, False
        ms = (time.perf_counter() - t0) * 1e3
        if not ok:
            self.Xg = np.zeros((4, self.N + 1))                        # discard a possibly bad warm start
            return None, {'ms': ms}
        X, U = sol.value(self.X), sol.value(self.U)
        self.Xg = np.column_stack([X[:, 1:], X[:, -1]])                # shift warm start by one step
        self.Ug = np.column_stack([U[:, 1:], U[:, -1]])
        return U[:, 0], {'ms': ms}
