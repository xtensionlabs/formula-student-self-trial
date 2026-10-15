#!/usr/bin/env python3
"""Score /perception/cones against the simulator's ground truth (no car motion needed).

  python3 scripts/eval_perception.py [--seconds 15] [--fov 90] [--range 20]

Ground truth: /fsds/testing_only/track (cones) + /fsds/testing_only/odom (true car pose). For every
perception message the true cones inside the LiDAR field of view are matched to detections within
`--match` metres. Reports recall per range bin, precision, position error and (if colours exist)
colour accuracy.  Needs the sim + bridge + cone_detector_node (+ TF); the car may stand still.
"""
import _paths  # noqa: F401  (sets up sys.path)
import argparse
import logging
import time
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from scipy.spatial import cKDTree
from fs_msgs.msg import Track
from fs_perception.msg import ConeArray

log = logging.getLogger('eval_perception')
FS_TO_OURS = {0: 2, 1: 1, 2: 3, 3: 3, 4: 0}      # fs_msgs colour enum -> our Cone.color (1 yellow 2 blue 3 orange)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=15.0)
    ap.add_argument('--fov', type=float, default=90.0, help='half field of view [deg] of the ground-truth region')
    ap.add_argument('--range', type=float, default=20.0)
    ap.add_argument('--match', type=float, default=0.6)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(name)s %(message)s')
    rclpy.init()
    node = rclpy.create_node('eval_perception')
    gt = {'cones': None, 'pose': None}
    bins = np.array([0, 5, 10, 15, 20, 30])
    hit, tot = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
    det_total, det_true, errs, col_ok, col_n = 0, 0, [], 0, 0

    def on_track(m: Track) -> None:
        gt['cones'] = np.array([[c.location.x, c.location.y, FS_TO_OURS.get(c.color, 0)] for c in m.track])

    def on_odom(m: Odometry) -> None:
        q = m.pose.pose.orientation
        gt['pose'] = np.array([m.pose.pose.position.x, m.pose.pose.position.y, 2 * np.arctan2(q.z, q.w)])

    def on_cones(m: ConeArray) -> None:
        nonlocal det_total, det_true, col_ok, col_n
        if gt['cones'] is None or gt['pose'] is None:
            return
        p = gt['pose']
        d = gt['cones'][:, :2] - p[:2]
        c, s = np.cos(p[2]), np.sin(p[2])
        rel = np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]])    # true cones, car frame
        r, bearing = np.hypot(rel[:, 0], rel[:, 1]), np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
        vis = (r < a.range) & (np.abs(bearing) < a.fov)
        det = np.array([[q.x, q.y, q.color] for q in m.cones]).reshape(-1, 3)
        tree = cKDTree(rel[vis]) if vis.any() else None
        det_total += len(det)
        if tree is not None and len(det):
            dd, ii = tree.query(det[:, :2])
            ok = dd < a.match
            det_true += int(ok.sum())
            errs.extend(dd[ok])
            known = ok & (det[:, 2] > 0)
            col_n += int(known.sum())
            col_ok += int((gt['cones'][vis][ii[known], 2] == det[known, 2]).sum())
        if vis.any():                                                      # recall: was each visible true cone detected?
            near = cKDTree(det[:, :2]).query(rel[vis])[0] < a.match if len(det) else np.zeros(vis.sum(), bool)
            idx = np.digitize(r[vis], bins) - 1
            for b, k in zip(idx, near):
                if 0 <= b < len(hit):
                    tot[b] += 1
                    hit[b] += k

    latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)   # the track may be published once
    node.create_subscription(Track, '/fsds/testing_only/track', on_track, latched)
    node.create_subscription(Odometry, '/fsds/testing_only/odom', on_odom, 1)
    node.create_subscription(ConeArray, '/perception/cones', on_cones, 10)
    end = time.time() + a.seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if gt['cones'] is None or gt['pose'] is None:
        log.error('no ground truth received (track=%s odom=%s): is the bridge running?', gt['cones'] is not None, gt['pose'] is not None)
    for lo, hi, h, t in zip(bins[:-1], bins[1:], hit, tot):
        log.info('recall %2d-%2d m : %5.1f%%  (%d true cone sightings)', lo, hi, 100 * h / max(t, 1), t)
    log.info('precision: %.1f%% of %d detections are real cones', 100 * det_true / max(det_total, 1), det_total)
    log.info('position error: mean %.3f m, p95 %.3f m', np.mean(errs) if errs else np.nan, np.percentile(errs, 95) if errs else np.nan)
    log.info('colour accuracy (coloured detections): %.1f%% of %d', 100 * col_ok / max(col_n, 1), col_n)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
