"""UKF predict/update tests (no ROS needed)."""
import numpy as np
from fs_state_estimation.ukf import UKF, process, h_imu, h_gss_world


def make() -> UKF:
    return UKF(np.zeros(6), np.eye(6) * 1e-4, np.eye(6) * 1e-3)


def test_predict_straight_line() -> None:
    # 10 m/s heading +x for 1 s -> ~10 m travelled, heading unchanged
    u = make(); u.x[3] = 10.0
    for _ in range(100):
        u.predict(0.0, 0.01)
    assert abs(u.x[0] - 10.0) < 0.05 and abs(u.x[2]) < 1e-6


def test_predict_grows_covariance() -> None:
    u = make(); p0 = u.P[0, 0]
    u.predict(0.0, 0.1)
    assert u.P[0, 0] > p0  # x variance grows with process noise


def test_update_pulls_state_and_shrinks_covariance() -> None:
    u = make(); u.P[2, 2] = 1.0; p0 = u.P[2, 2]  # loose heading prior
    u.update(np.array([0.3, 0.0]), h_imu, np.eye(2) * 1e-3, z_angle_idx=[0])
    assert 0.25 < u.x[2] <= 0.3 and u.P[2, 2] < p0


def test_angle_wraparound() -> None:
    # state at +3.1 rad, measurement at -3.1 rad: true difference is ~0.083 rad, not 6.2
    u = make(); u.x[2] = 3.1
    u.update(np.array([-3.1, 0.0]), h_imu, np.eye(2) * 1e-3, z_angle_idx=[0])
    assert abs(abs(u.x[2]) - np.pi) < 0.1


def test_gss_world_rotation() -> None:
    # facing +y (yaw=90deg) with body vx=5 -> world velocity (0, 5)
    assert np.allclose(h_gss_world(np.array([0, 0, np.pi / 2, 5, 0, 0])), [0, 5], atol=1e-9)


def test_process_bicycle_lateral_velocity() -> None:
    assert abs(process(np.array([0, 0, 0, 5, 0, 1.0]), 0, 0.01)[4] - 0.765) < 1e-9
