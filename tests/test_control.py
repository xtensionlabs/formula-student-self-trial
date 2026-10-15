"""MPC cost function, path features and closed-loop controller tests (no ROS needed)."""
import os
import numpy as np
from fs_control.mpc import Mpc, MpcWeights, VehicleLimits, make_dynamics, make_stage_cost
from fs_control.reference import PathRef
from fs_control.sim import random_track, run

W = MpcWeights()
COST = make_stage_cost()


def stage(x, r, u=(0.0, 0.0), up=(0.0, 0.0), w=W) -> float:
    return float(COST(np.array(x), np.array(r), np.array(u), np.array(up), w.vec()))


def test_cost_zero_on_reference() -> None:
    assert stage([5, 2, 0.3, 8], [5, 2, 0.3, 8]) == 0.0


def test_cost_cross_track_term() -> None:
    # reference heading +x; car 1 m to the left -> only the cross-track weight applies
    assert abs(stage([0, 1, 0, 5], [0, 0, 0, 5]) - W.q_ct) < 1e-9


def test_cost_heading_speed_and_input_terms() -> None:
    assert abs(stage([0, 0, np.pi / 2, 5], [0, 0, 0, 5]) - 2 * W.q_head) < 1e-9     # 90 deg heading error
    assert abs(stage([0, 0, 0, 7], [0, 0, 0, 5]) - 4 * W.q_v) < 1e-9                 # 2 m/s speed error
    assert abs(stage([0, 0, 0, 5], [0, 0, 0, 5], u=(0.1, 0), up=(0.0, 0)) - (W.r_d + W.r_dd) * 0.01) < 1e-9


def test_cost_is_wrap_free() -> None:
    # heading 2*pi + 0.1 vs 0.1 must cost the same as identical headings
    assert stage([0, 0, 2 * np.pi + 0.1, 5], [0, 0, 0.1, 5]) < 1e-12


def test_dynamics_straight_and_turning() -> None:
    step = make_dynamics(VehicleLimits(), 0.05)
    assert np.allclose(np.array(step([0, 0, 0, 10], [0, 0])).ravel(), [0.5, 0, 0, 10])
    x = np.array(step([0, 0, 0, 10], [0.2, 0])).ravel()
    assert x[1] > 0 and x[2] > 0                                  # positive steering turns left


def test_path_features_signed_cte() -> None:
    xy = np.column_stack([np.arange(0, 100, 0.5), np.zeros(200)])
    ref = PathRef(xy, np.full(200, 8.0))
    f = ref.features(np.array([20.0, 0.5, 0.1]), 6.0)             # 0.5 m left of the path, slightly rotated
    assert abs(f[0] - 0.5) < 1e-6 and abs(f[1] - 0.1) < 1e-6 and abs(f[2] + 2.0) < 1e-6 and abs(f[3]) < 1e-6


def test_mpc_tracks_random_track() -> None:
    mpc = Mpc()
    from fs_control.sim import mpc_controller
    r = run(mpc_controller(mpc, W), random_track(np.random.default_rng(3)), T=10)
    assert not r['off_track'] and np.sqrt(np.mean(r['feat'][:, 0] ** 2)) < 0.15


def test_ann_torchscript_outputs_within_limits() -> None:
    import pytest
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ann_controller.pt')
    if not os.path.exists(path):
        pytest.skip('train with scripts/train_ann_controller.py first')
    import torch
    out = torch.jit.load(path)(torch.tensor([[0.3, 0.1, -1.0, 0.05], [5.0, 2.0, 9.0, 0.5]]))
    assert out.shape == (2, 2) and (out[:, 0].abs() <= 0.436 + 1e-6).all() and ((out[:, 1] >= -10) & (out[:, 1] <= 5)).all()
