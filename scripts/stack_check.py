#!/usr/bin/env python3
"""Smoke check of a running stack: listens for a few seconds and reports what each key topic carried.

  python3 scripts/stack_check.py [--seconds 8]
"""
import _paths  # noqa: F401  (sets up sys.path)
import argparse
import logging
import time

import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, ReliabilityPolicy
from diagnostic_msgs.msg import DiagnosticArray
from sensor_msgs.msg import Image, Imu, PointCloud2
from geometry_msgs.msg import TwistWithCovarianceStamped
from fs_msgs.msg import ControlCommand
from fs_perception.msg import ConeArray, SpeedProfile
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path

log = logging.getLogger('stack_check')
LATCHED = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=8.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(name)s %(message)s')
    rclpy.init()
    node = rclpy.create_node('stack_check')
    seen, lat = {}, {}

    def watch(msg_type, topic, summary, qos=10) -> None:
        def cb(m) -> None:
            n = seen.get(topic, (0, ''))[0] + 1
            seen[topic] = (n, summary(m))
        node.create_subscription(msg_type, topic, cb, qos)

    def diag(m: DiagnosticArray) -> None:
        for s in m.status:
            lat[s.name] = s.values[0].value
    node.create_subscription(DiagnosticArray, '/diagnostics', diag, 50)
    sensor = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)      # listen like the nodes do
    watch(Imu, '/fsds/imu', lambda m: f'yaw_q=({m.orientation.z:.2f},{m.orientation.w:.2f})', sensor)
    watch(TwistWithCovarianceStamped, '/fsds/gss', lambda m: f'v=({m.twist.twist.linear.x:.1f},{m.twist.twist.linear.y:.1f})', sensor)
    watch(PointCloud2, '/fsds/lidar/Lidar1', lambda m: f'{m.width * m.height} pts frame={m.header.frame_id}', sensor)
    watch(Image, '/fsds/camera/cam1', lambda m: f'{m.width}x{m.height} {m.encoding}', sensor)
    watch(Odometry, '/state/odometry', lambda m: f'x={m.pose.pose.position.x:.1f} v={m.twist.twist.linear.x:.1f}')
    watch(ConeArray, '/perception/cones', lambda m: f'{len(m.cones)} cones')
    watch(ConeArray, '/slam/map', lambda m: f'{len(m.cones)} landmarks')
    watch(ConeArray, '/slam/map_corrected', lambda m: f'{len(m.cones)} landmarks', LATCHED)
    watch(PoseWithCovarianceStamped, '/slam/pose', lambda m: f'({m.pose.pose.position.x:.1f}, {m.pose.pose.position.y:.1f})')
    watch(Path, '/planning/centerline', lambda m: f'{len(m.poses)} poses')
    watch(SpeedProfile, '/planning/velocity_profile', lambda m: f'v {min(m.speed):.1f}..{max(m.speed):.1f} m/s')
    watch(ControlCommand, '/fsds/control_command', lambda m: f'steer={m.steering:.2f} thr={m.throttle:.2f} brk={m.brake:.2f}')
    end = time.time() + a.seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    for topic in ('/fsds/imu', '/fsds/gss', '/fsds/lidar/Lidar1', '/fsds/camera/cam1', '/state/odometry', '/perception/cones', '/slam/pose', '/slam/map', '/slam/map_corrected',
                  '/planning/centerline', '/planning/velocity_profile', '/fsds/control_command'):
        n, last = seen.get(topic, (0, ''))
        log.info('%-28s %5.1f Hz  %s', topic, n / a.seconds, last or 'NO DATA')
    log.info('node latencies [ms]: %s', ', '.join(f'{k}={v}' for k, v in sorted(lat.items())) or 'none')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
