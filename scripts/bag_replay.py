#!/usr/bin/env python3
"""Offline testing without FSDS.

  bag_replay.py record  my_run              # while FSDS + bridge run: record the sensor topics
  bag_replay.py play    my_run [--rate 1.0] # replay the bag (loops) -> run the stack on recorded data
  bag_replay.py synthetic [--laps 3]        # no bag at all: publish /perception/cones + /state/odometry
                                            # from a simulated circular track with drifting odometry

`record`/`play` wrap `ros2 bag`; the stack's own topics are not recorded so a replay recomputes them.
`synthetic` feeds the stack from the cone level, so it exercises slam, loop closure, planning, RViz
(and the controllers' inputs) without perception, state estimation or the simulator.
"""
import _paths  # noqa: F401  (sets up sys.path)
import argparse
import logging
import math
import os
import subprocess
import sys
import time

log = logging.getLogger('bag_replay')
# Raw sensor topics of the FSDS bridge (names follow D:/FSDS/settings.json) + static TF for the extrinsics
SENSOR_TOPICS = ['/fsds/imu', '/fsds/gss', '/fsds/lidar/Lidar1', '/fsds/camera/cam1', '/tf_static',
                 '/fsds/testing_only/odom']


def synthetic(laps: float, hz: float) -> None:
    """Publish a simulated drive as ConeArray + Odometry messages (see tests/track_sim.py)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
    import rclpy
    from nav_msgs.msg import Odometry
    from fs_perception.msg import Cone, ConeArray
    from track_sim import drive

    rclpy.init()
    node = rclpy.create_node('synthetic_drive')
    cones_pub = node.create_publisher(ConeArray, '/perception/cones', 10)
    odom_pub = node.create_publisher(Odometry, '/state/odometry', 10)
    steps = drive(laps=laps, step=0.5)

    while rclpy.ok():
        try:
            _, odom, xy, col, cov = next(steps)
        except StopIteration:
            log.info('synthetic drive finished')
            break
        now = node.get_clock().now().to_msg()
        o = Odometry()
        o.header.stamp, o.header.frame_id = now, 'odom'
        o.pose.pose.position.x, o.pose.pose.position.y = float(odom[0]), float(odom[1])
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(odom[2] / 2), math.cos(odom[2] / 2)
        o.twist.twist.linear.x = 0.5 * hz
        odom_pub.publish(o)
        m = ConeArray()
        m.header.stamp, m.header.frame_id = now, 'fsds/FSCar'
        m.cones = [Cone(x=float(p[0]), y=float(p[1]), color=int(c), confidence=0.9,
                        covariance=[float(v) for v in cv.ravel()]) for p, c, cv in zip(xy, col, cov)]
        cones_pub.publish(m)
        time.sleep(1.0 / hz)                            # plain sleep: no executor is spinning
    node.destroy_node()
    rclpy.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('record'); r.add_argument('name')
    p = sub.add_parser('play'); p.add_argument('name'); p.add_argument('--rate', type=float, default=1.0)
    s = sub.add_parser('synthetic'); s.add_argument('--laps', type=float, default=3.0); s.add_argument('--hz', type=float, default=20.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')

    if a.cmd == 'record':
        subprocess.run(['ros2', 'bag', 'record', '-o', a.name, *SENSOR_TOPICS], check=True)
    elif a.cmd == 'play':
        subprocess.run(['ros2', 'bag', 'play', a.name, '--loop', '--rate', str(a.rate)], check=True)
    else:
        synthetic(a.laps, a.hz)


if __name__ == '__main__':
    main()
