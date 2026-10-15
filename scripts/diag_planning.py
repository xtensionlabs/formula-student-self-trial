#!/usr/bin/env python3
"""Snapshot of what the planner works from, scored against ground truth (car may stand still).

  python3 scripts/diag_planning.py [--seconds 8]

Prints: SLAM map size, colour accuracy of the map cones (vs the sim's true colours), the planned path
(length, endpoints) and, for map cones in front of the car, assigned vs true colour.
"""
import _paths  # noqa: F401  (sets up sys.path)
import argparse
import logging
import time

import numpy as np
import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from nav_msgs.msg import Odometry, Path
from scipy.spatial import cKDTree
from fs_msgs.msg import Track
from fs_perception.msg import ConeArray

log = logging.getLogger('diag_planning')
FS_TO_OURS = {0: 2, 1: 1, 2: 3, 3: 3, 4: 0}
NAME = {0: 'unk', 1: 'yel', 2: 'blu', 3: 'org'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=8.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(name)s %(message)s')
    rclpy.init()
    node = rclpy.create_node('diag_planning')
    s = {'gt': None, 'pose': None, 'map': None, 'path': None}
    latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(Track, '/fsds/testing_only/track', lambda m: s.update(
        gt=np.array([[c.location.x, c.location.y, FS_TO_OURS.get(c.color, 0)] for c in m.track])), latched)
    node.create_subscription(Odometry, '/fsds/testing_only/odom', lambda m: s.update(
        pose=np.array([m.pose.pose.position.x, m.pose.pose.position.y, 2 * np.arctan2(m.pose.pose.orientation.z, m.pose.pose.orientation.w)])), 1)
    node.create_subscription(ConeArray, '/slam/map', lambda m: s.update(map=np.array([[c.x, c.y, c.color] for c in m.cones]).reshape(-1, 3)), 1)
    node.create_subscription(Path, '/planning/centerline', lambda m: s.update(path=np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses])), 1)
    end = time.time() + a.seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if any(v is None for v in s.values()):
        log.error('missing data: %s', {k: v is not None for k, v in s.items()})
        return
    gt, pose, m = s['gt'], s['pose'], s['map']
    log.info('car (ground truth) x=%.1f y=%.1f yaw=%.0f deg; map has %d cones', pose[0], pose[1], np.degrees(pose[2]), len(m))
    d, i = cKDTree(gt[:, :2]).query(m[:, :2])
    ok = d < 0.7
    log.info('map cones matching a real cone (<0.7 m): %d/%d  (mean error %.2f m)', ok.sum(), len(m), d[ok].mean() if ok.any() else np.nan)
    known = ok & (m[:, 2] > 0)
    log.info('colour correct: %d/%d coloured map cones', (gt[i[known], 2] == m[known, 2]).sum(), known.sum())
    c, sn = np.cos(pose[2]), np.sin(pose[2])
    rel = np.column_stack([c * (m[:, 0] - pose[0]) + sn * (m[:, 1] - pose[1]), -sn * (m[:, 0] - pose[0]) + c * (m[:, 1] - pose[1])])
    order = np.argsort(rel[:, 0])
    log.info('map cones ahead of the car (car frame x fwd, y left): assigned vs true colour')
    for k in order[rel[order, 0] > -1][:14]:
        log.info('   (%5.1f,%5.1f)  assigned=%s  true=%s%s', rel[k, 0], rel[k, 1], NAME[int(m[k, 2])],
                 NAME[int(gt[i[k], 2])] if ok[k] else '--', '' if (not ok[k] or gt[i[k], 2] == m[k, 2]) else '   <-- WRONG')
    p = s['path']
    log.info('planned path: %d poses, %.1f m, from (%.1f,%.1f) to (%.1f,%.1f)', len(p),
             np.sum(np.hypot(*np.diff(p, axis=0).T)) if len(p) > 1 else 0, *p[0], *p[-1])
    rclpy.shutdown()


if __name__ == '__main__':
    main()
