#!/usr/bin/env python3
"""Compare the time bases of the simulator topics (header stamps) with wall time and with the car's real motion.

  python3 scripts/clock_check.py [--seconds 8]

If the sim runs slower than real time, stamps and physics disagree and any filter that integrates
stamp differences (our UKF) mis-scales position. Prints stamp-seconds per wall-second for each topic.
"""
import _paths  # noqa: F401  (sets up sys.path)
import argparse
import logging
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2

log = logging.getLogger('clock_check')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=8.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    rclpy.init()
    node = rclpy.create_node('clock_check')
    rec = {}

    def watch(msg_type, topic, qos) -> None:
        def cb(m) -> None:
            rec.setdefault(topic, []).append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, time.time()))
        node.create_subscription(msg_type, topic, cb, qos)

    watch(Imu, '/fsds/imu', qos_profile_sensor_data)
    watch(TwistWithCovarianceStamped, '/fsds/gss', qos_profile_sensor_data)
    watch(PointCloud2, '/fsds/lidar/Lidar1', qos_profile_sensor_data)
    watch(Odometry, '/fsds/testing_only/odom', 1)
    end = time.time() + a.seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    for topic, r in rec.items():
        r = np.array(r)
        if len(r) < 3:
            continue
        dstamp, dwall = r[-1, 0] - r[0, 0], r[-1, 1] - r[0, 1]
        dts = np.diff(r[:, 0])
        log.info('%-26s %5.1f msgs/wall-s | stamp advances %.2f s per wall-s | stamp dt mean %.4f s (max %.3f)',
                 topic, len(r) / dwall, dstamp / dwall, dts.mean(), dts.max())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
