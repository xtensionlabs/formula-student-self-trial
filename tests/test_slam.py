"""EKF-SLAM and pose-graph loop-closure tests on a synthetic circular track."""
import numpy as np
from scipy.spatial import cKDTree
from fs_slam.ekf_slam import EkfSlam, relative_pose
from fs_slam.pose_graph import PoseGraph
from track_sim import drive, make_cones, to_frame


def test_ekf_slam_builds_consistent_map() -> None:
    slam, last = EkfSlam(), None
    start = None
    for true, odom, xy, col, cov in drive(laps=1.1):
        start = true if start is None else start
        if last is not None:
            slam.predict(relative_pose(last, odom))
        last = odom
        slam.update(xy, col, np.ones(len(xy)), cov)
    pos, colors = slam.landmarks()
    cones, true_col = make_cones()
    truth = to_frame(start, cones)                       # ground truth in the start frame (= map frame)
    d, k = cKDTree(truth).query(pos)
    assert len(pos) <= len(truth) * 1.15                 # no landmark explosion
    assert np.median(d) < 0.5                            # map close to truth despite biased odometry
    assert (colors == true_col[k]).mean() > 0.95         # colours kept in separate maps


def test_loop_closure_reduces_drift() -> None:
    g, start, truth, raw = PoseGraph(), None, [], []
    for true, odom, xy, col, _ in drive(laps=1.25):
        start = true if start is None else start
        if g.needs_keyframe(odom):                       # odometry starts at the origin = map frame
            g.add_keyframe(odom, xy, col)
            truth.append(relative_pose(start, true)[:2])  # ground-truth keyframe position in map frame
            raw.append(odom[:2])
    assert g.n_loops > 0
    truth, raw = np.array(truth), np.array(raw)
    opt = np.array([kf.pose[:2] for kf in g.kfs])
    err_raw = np.linalg.norm(raw - truth, axis=1).mean()
    err_opt = np.linalg.norm(opt - truth, axis=1).mean()
    assert err_opt < 0.5 * err_raw                       # loop closure removes most of the drift
