"""Delaunay centerline + speed-profile tests on the synthetic circular track."""
import numpy as np
from fs_planning.centerline import build_centerline, midpoints, CenterlineParams, sample
from fs_planning.speed_gp import SpeedGP, limit_accel
from track_sim import HALF_W, R, make_cones


def test_midpoints_lie_on_centerline() -> None:
    pos, col = make_cones()
    mids = midpoints(pos, col, CenterlineParams())
    assert len(mids) > 40
    assert np.all(np.abs(np.hypot(mids[:, 0], mids[:, 1]) - R) < 0.6)   # only yellow<->blue edges


def test_closed_full_track() -> None:
    pos, col = make_cones()
    car = np.array([R, 0.0, np.pi / 2])                   # on the centreline, heading counter-clockwise
    spline, length, closed = build_centerline(pos, col, car)
    s, xy, kappa, _ = sample(spline, length, 0.5, closed)
    assert closed and abs(length - 2 * np.pi * R) < 3.0
    assert np.all(np.abs(np.hypot(xy[:, 0], xy[:, 1]) - R) < 0.5)
    assert abs(np.median(kappa) - 1 / R) < 0.01           # curvature of a circle = 1/R


def test_open_track_ahead_only() -> None:
    pos, col = make_cones()
    ang = np.arctan2(pos[:, 1], pos[:, 0])
    keep = (ang > -0.1) & (ang < 1.5)                     # only the first ~90 degrees are mapped
    car = np.array([R, 0.0, np.pi / 2])
    spline, length, closed = build_centerline(pos[keep], col[keep], car)
    assert not closed and 15 < length < 30
    assert spline(0.0)[0] > R * 0.9                       # starts near the car, goes forward


def test_speed_gp_follows_physics_and_learns() -> None:
    gp = SpeedGP(a_lat=15.0, v_cap=12.0)
    mean, _ = gp.predict(np.array([0.0, 0.1, 0.4]))
    assert mean[0] > 11 and abs(mean[1] - np.sqrt(15 / 0.1)) < 1.0 and mean[2] < 7
    for _ in range(5):                                    # repeated evidence that 0.1 1/m is only safe at 9 m/s
        gp.update(np.array([0.1]), np.array([9.0]))
    assert gp.predict(np.array([0.1]))[0][0] < mean[1]


def test_limit_accel_respects_braking() -> None:
    s = np.arange(0, 100, 1.0)
    v = np.full_like(s, 12.0); v[60] = 3.0                # one tight corner
    out = limit_accel(s, v, a_acc=5.0, a_brake=8.0, closed=False)
    assert out[60] == 3.0 and out[59] <= np.sqrt(3 ** 2 + 2 * 8) + 1e-9 and np.all(out <= v + 1e-9)
