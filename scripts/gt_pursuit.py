#!/usr/bin/env python3
"""Plant check: drive with pure pursuit on the simulator's GROUND-TRUTH pose and track (no perception/SLAM/MPC).

  python3 scripts/gt_pursuit.py [--speed 3] [--seconds 25] [--gain 1.0]

If this follows the centerline well, the car/command interface is fine and any problem is in our pipeline;
if it oscillates, the cause is the plant (steering gain, latency, simulator frame rate). Stack nodes must NOT
be running (they would also command the car). Aborts and brakes if the car leaves the track (>2.5 m).
"""
import _paths  # noqa: F401  (sets up sys.path)
import argparse
import logging
import time

import numpy as np
import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from nav_msgs.msg import Odometry
from fs_msgs.msg import ControlCommand, Track
from fs_planning.centerline import build_centerline, sample

log = logging.getLogger('gt_pursuit')
FS_TO_OURS = {0: 2, 1: 1, 2: 3, 3: 3, 4: 0}
L, D_MAX = 1.53, 0.436


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=float, default=3.0)
    ap.add_argument('--seconds', type=float, default=25.0)
    ap.add_argument('--gain', type=float, default=1.0, help='multiplies the steering command (tests steering-gain error)')
    ap.add_argument('--compare', action='store_true',
                    help='also report our estimated pose / planned path error (needs the stack with control:=false)')
    ap.add_argument('--lookahead', type=float, default=0.8, help='lookahead seconds')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    rclpy.init()
    node = rclpy.create_node('gt_pursuit')
    s = {'path': None, 'pose': None, 'v': 0.0, 'trace': []}
    latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

    def on_track(m: Track) -> None:
        if s['path'] is not None or not m.track:
            return
        pos = np.array([[c.location.x, c.location.y] for c in m.track])
        col = np.array([FS_TO_OURS.get(c.color, 0) for c in m.track])
        res = build_centerline(pos, col, np.zeros(3))
        if res:
            s['path'] = sample(res[0], res[1], 0.5, res[2])[1]

    def on_odom(m: Odometry) -> None:
        q = m.pose.pose.orientation
        s['pose'] = np.array([m.pose.pose.position.x, m.pose.pose.position.y, 2 * np.arctan2(q.z, q.w)])
        s['v'] = float(np.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y))
        s['trace'].append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, time.time(), *s['pose'][:2], s['v']))

    node.create_subscription(Track, '/fsds/testing_only/track', on_track, latched)
    node.create_subscription(Odometry, '/fsds/testing_only/odom', on_odom, 1)
    pub = node.create_publisher(ControlCommand, '/fsds/control_command', 10)
    inputs = None
    if a.compare:                                                          # our own state estimate + planned path
        from fs_control.ctrl_common import ControlInputs, declare_common
        declare_common(node)
        inputs = ControlInputs(node)
        from geometry_msgs.msg import PoseWithCovarianceStamped
        node.create_subscription(Odometry, '/state/odometry', lambda m: s.update(
            ukf=np.array([m.pose.pose.position.x, m.pose.pose.position.y, 2 * np.arctan2(m.pose.pose.orientation.z, m.pose.pose.orientation.w)])), 5)
        node.create_subscription(PoseWithCovarianceStamped, '/slam/pose', lambda m: s.update(
            slam=np.array([m.pose.pose.position.x, m.pose.pose.position.y, 2 * np.arctan2(m.pose.pose.orientation.z, m.pose.pose.orientation.w)])), 5)
    t0 = time.time()
    while (s['path'] is None or s['pose'] is None) and time.time() - t0 < 10:
        rclpy.spin_once(node, timeout_sec=0.1)
    path, last, tlog, cte_hist = s['path'], 0, 0.0, []
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        rclpy.spin_once(node, timeout_sec=0.02)
        x, y, yaw = s['pose']
        n = len(path)
        idx = (np.arange(last - 5, last + 40) % n)
        last = int(idx[np.argmin(np.linalg.norm(path[idx] - s['pose'][:2], axis=1))])
        cte = float(np.linalg.norm(path[last] - s['pose'][:2]))
        ld = max(2.0, a.lookahead * s['v'])                                 # lookahead distance
        j = last
        while np.linalg.norm(path[j % n] - s['pose'][:2]) < ld and j < last + 60:
            j += 1
        tgt = path[j % n]
        alpha = np.arctan2(tgt[1] - y, tgt[0] - x) - yaw
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        delta = np.arctan2(2 * L * np.sin(alpha), ld)                        # pure pursuit steering angle (left +)
        cmd = ControlCommand()
        cmd.steering = float(np.clip(-a.gain * delta / D_MAX, -1, 1))        # FSDS: +1 = right
        err = a.speed - s['v']
        cmd.throttle, cmd.brake = (float(np.clip(0.3 * err, 0, 1)), 0.0) if err > 0 else (0.0, float(np.clip(-0.3 * err, 0, 1)))
        pub.publish(cmd)
        cte_hist.append(cte)
        if time.time() - tlog > 1.0:
            tlog = time.time()
            extra = ''
            if inputs is not None and inputs.state() is not None:
                est = inputs.state()
                for name in ('ukf', 'slam'):                                # error of each stage on its own
                    if name in s:
                        extra += ' %s err %.2f m (x %+.1f vs gt %+.1f);' % (name, np.hypot(*(s[name][:2] - s['pose'][:2])), s[name][0], s['pose'][0])
                yerr = np.degrees(np.arctan2(np.sin(est[2] - yaw), np.cos(est[2] - yaw)))
                extra += ' | est pos err %.2f m, yaw err %+.1f deg, v err %+.2f' % (np.hypot(*(est[:2] - s['pose'][:2])), yerr, est[3] - s['v'])
                if inputs.ref is not None:                                  # distance of our planned path points to the ideal path
                    d = np.min(np.linalg.norm(inputs.ref.xy[::2, None, :] - path[None, ::2, :], axis=2), axis=1)
                    extra += ', plan: %d pts, err mean %.2f max %.2f m' % (len(inputs.ref.xy), d.mean(), d.max())
            log.info('t=%4.1f pos=(%6.1f,%6.1f) v=%4.1f steer_cmd=%5.2f off=%.2f%s', tlog - t0, x, y, s['v'], cmd.steering, cte, extra)
        if cte > 2.5:
            log.info('off track (%.1f m): aborting', cte)
            break
    end = time.time() + 1.5
    while time.time() < end:
        pub.publish(ControlCommand(brake=1.0))
        rclpy.spin_once(node, timeout_sec=0.05)
    tr = np.array(s['trace'])                                              # (stamp, wall, x, y, reported speed)
    tr = tr[tr[:, 4] > 1.0]                                                # only while driving
    if len(tr) > 10:
        dist = np.sum(np.hypot(np.diff(tr[:, 2]), np.diff(tr[:, 3])))
        log.info('clock check while driving: distance %.1f m | reported speed mean %.2f m/s | distance / stamp-time = %.2f m/s | '
                 'distance / wall-time = %.2f m/s', dist, tr[:, 4].mean(), dist / (tr[-1, 0] - tr[0, 0]), dist / (tr[-1, 1] - tr[0, 1]))
    log.info('result: mean off %.2f m, max %.2f m over %d samples', np.mean(cte_hist), np.max(cte_hist), len(cte_hist))
    rclpy.shutdown()


if __name__ == '__main__':
    main()
