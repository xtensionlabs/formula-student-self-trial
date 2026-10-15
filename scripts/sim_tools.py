#!/usr/bin/env python3
"""Small simulator helpers (need the bridge).

  sim_tools.py reset                   put the car back at the start (bridge service /fsds/reset)
  sim_tools.py steer 0.4 [--seconds 2] drive gently with a constant steering command and report which way the
                                       car turned (checks the steering sign convention), then brake
"""
import _paths  # noqa: F401  (sets up sys.path)
import argparse
import logging
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from fs_msgs.msg import ControlCommand
from fs_msgs.srv import Reset

log = logging.getLogger('sim_tools')


def brake(node, pub, seconds: float = 1.5) -> None:
    end = time.time() + seconds
    while time.time() < end:
        pub.publish(ControlCommand(brake=1.0))
        rclpy.spin_once(node, timeout_sec=0.05)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['reset', 'steer'])
    ap.add_argument('value', nargs='?', type=float, default=0.4)
    ap.add_argument('--seconds', type=float, default=2.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(name)s %(message)s')
    rclpy.init()
    node = rclpy.create_node('sim_tools')
    pub = node.create_publisher(ControlCommand, '/fsds/control_command', 10)
    if a.cmd == 'reset':
        cli = node.create_client(Reset, '/fsds/reset')
        if not cli.wait_for_service(timeout_sec=10):
            log.error('no /fsds/reset service: bridge running?')
            return
        rclpy.spin_until_future_complete(node, cli.call_async(Reset.Request(wait_on_last_task=False)), timeout_sec=20)
        log.info('reset done')
    else:
        pose = []
        node.create_subscription(Odometry, '/fsds/testing_only/odom', lambda m: pose.append(
            (m.pose.pose.position.x, m.pose.pose.position.y, 2 * np.arctan2(m.pose.pose.orientation.z, m.pose.pose.orientation.w))), 1)
        t0 = time.time()
        while not pose and time.time() - t0 < 5:
            rclpy.spin_once(node, timeout_sec=0.1)
        start = pose[-1]
        end = time.time() + a.seconds
        while time.time() < end:                                   # gentle: throttle 0.25 keeps speed low
            pub.publish(ControlCommand(throttle=0.25, steering=a.value))
            rclpy.spin_once(node, timeout_sec=0.05)
        stop = pose[-1]
        brake(node, pub)
        dyaw = np.degrees(np.arctan2(np.sin(stop[2] - start[2]), np.cos(stop[2] - start[2])))
        log.info('steering command %+.2f -> yaw changed %+.1f deg, moved dx=%.1f dy=%+.1f m  => %s',
                 a.value, dyaw, stop[0] - start[0], stop[1] - start[1],
                 'turned LEFT (ENU yaw increases)' if dyaw > 2 else 'turned RIGHT' if dyaw < -2 else 'no clear turn')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
