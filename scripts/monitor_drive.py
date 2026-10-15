#!/usr/bin/env python3
"""Live drive monitor: one line per second + CSV, using the simulator's ground truth.

  python3 scripts/monitor_drive.py [--seconds 120] [--csv /tmp/drive.csv]

Columns: t, ground-truth x/y, speed, commands, our map/path sizes, cones knocked (doo), laps, and
`off` = distance of the true car position from the ideal centerline built from the ground-truth
track (m, lower is better), `v_ref` = planner's target speed at the car.
"""
import _paths  # noqa: F401  (sets up sys.path)
import argparse
import csv
import logging
import time

import numpy as np
import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from nav_msgs.msg import Odometry, Path
from fs_msgs.msg import ControlCommand, ExtraInfo, Track
from fs_perception.msg import ConeArray, SpeedProfile
from fs_planning.centerline import build_centerline, sample

log = logging.getLogger('monitor')
FS_TO_OURS = {0: 2, 1: 1, 2: 3, 3: 3, 4: 0}      # fs_msgs colour -> ours (1 yellow, 2 blue, 3 orange)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=120.0)
    ap.add_argument('--csv', default='/tmp/drive.csv')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    rclpy.init()
    node = rclpy.create_node('monitor_drive')
    s = {'pose': None, 'v': 0.0, 'cmd': (0, 0, 0), 'map': 0, 'path': 0, 'doo': 0, 'laps': [], 'vref': np.nan,
         'ideal': None, 'cones': 0}
    latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

    def on_track(m: Track) -> None:
        if s['ideal'] is not None or not m.track:
            return
        pos = np.array([[c.location.x, c.location.y] for c in m.track])
        col = np.array([FS_TO_OURS.get(c.color, 0) for c in m.track])
        res = build_centerline(pos, col, np.array([0.0, 0.0, 0.0]))     # ideal path from true colours
        if res:
            s['ideal'] = sample(res[0], res[1], 0.5, res[2])[1]
            log.info('ground-truth track: %d cones, ideal centerline %.0f m', len(pos), res[1])

    def on_odom(m: Odometry) -> None:
        q = m.pose.pose.orientation
        s['pose'] = np.array([m.pose.pose.position.x, m.pose.pose.position.y])
        s['v'] = float(np.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y))

    node.create_subscription(Track, '/fsds/testing_only/track', on_track, latched)
    node.create_subscription(Odometry, '/fsds/testing_only/odom', on_odom, 1)
    node.create_subscription(ControlCommand, '/fsds/control_command', lambda m: s.update(cmd=(m.steering, m.throttle, m.brake)), 5)
    node.create_subscription(ConeArray, '/slam/map', lambda m: s.update(map=len(m.cones)), 5)
    node.create_subscription(ConeArray, '/perception/cones', lambda m: s.update(cones=len(m.cones)), 5)
    node.create_subscription(Path, '/planning/centerline', lambda m: s.update(path=len(m.poses)), 5)
    node.create_subscription(SpeedProfile, '/planning/velocity_profile', lambda m: s.update(vref=min(m.speed[:10]) if m.speed else np.nan), 5)
    node.create_subscription(ExtraInfo, '/fsds/testing_only/extra_info', lambda m: s.update(doo=m.doo_counter, laps=list(m.laps)), 5)

    t0, last = time.time(), 0.0
    with open(a.csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'x', 'y', 'v', 'steer', 'throttle', 'brake', 'cones', 'map', 'path', 'v_ref', 'doo', 'laps', 'off'])
        while time.time() - t0 < a.seconds and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.time() - last < 1.0 or s['pose'] is None:
                continue
            last = time.time()
            off = float(np.min(np.linalg.norm(s['ideal'] - s['pose'], axis=1))) if s['ideal'] is not None else np.nan
            row = [round(last - t0, 1), *np.round(s['pose'], 1), round(s['v'], 1), *np.round(s['cmd'], 2), s['cones'], s['map'],
                   s['path'], round(s['vref'], 1), s['doo'], len(s['laps']), round(off, 2)]
            w.writerow(row)
            log.info('t=%5.1f pos=(%6.1f,%6.1f) v=%4.1f cmd[st %5.2f th %4.2f br %4.2f] cones=%2d map=%3d path=%3d vref=%4.1f doo=%d laps=%d off=%.2f',
                     *row)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
