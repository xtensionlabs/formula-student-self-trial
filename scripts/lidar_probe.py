#!/usr/bin/env python3
"""Inspect one LiDAR scan: ring structure, ground plane height, and what the cone detector makes of it.

  python3 scripts/lidar_probe.py /fsds/lidar/Lidar1
"""
import _paths  # noqa: F401  (sets up sys.path)
import logging
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from fs_perception_py.cone_detector_node import cloud_to_xyz
from fs_perception_py.lidar_cones import LidarParams, detect_cones, fit_ground

log = logging.getLogger('lidar_probe')


def main() -> None:
    topic = sys.argv[1] if len(sys.argv) > 1 else '/fsds/lidar/Lidar1'
    logging.basicConfig(level=logging.INFO, format='%(name)s %(message)s')
    rclpy.init()
    node = rclpy.create_node('lidar_probe')
    got = []
    node.create_subscription(PointCloud2, topic, got.append, qos_profile_sensor_data)
    end = time.time() + 10
    while not got and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not got:
        log.error('no scan on %s', topic)
        return
    pts = cloud_to_xyz(got[0])
    p = LidarParams()
    log.info('%s frame=%s  %d points  fields=%s', topic, got[0].header.frame_id, len(pts), [f.name for f in got[0].fields])
    elev = np.degrees(np.arctan2(pts[:, 2], np.hypot(pts[:, 0], pts[:, 1])))
    rings, counts = np.unique(np.round(elev, 0), return_counts=True)
    log.info('elevation rings [deg: points] %s', dict(zip(rings.tolist(), counts.tolist())))
    log.info('x %.1f..%.1f  y %.1f..%.1f  z %.2f..%.2f', pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(),
             pts[:, 1].max(), pts[:, 2].min(), pts[:, 2].max())
    plane = fit_ground(pts[np.hypot(pts[:, 0], pts[:, 1]) < p.max_range], p, np.random.default_rng(0))
    if plane is not None:
        n_in = int((np.abs(pts @ plane[:3] + plane[3]) < p.ground_dist).sum())
        log.info('RANSAC ground: normal=(%.2f,%.2f,%.2f) height below sensor=%.2f m, %d inliers', *plane[:3], -plane[3] / plane[2], n_in)
    else:
        log.warning('RANSAC found no ground plane')
    c, _ = detect_cones(pts, p)
    log.info('cones found with default params: %d  %s', len(c), np.round(c[:8, :2], 1).tolist())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
