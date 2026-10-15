#!/usr/bin/env python3
"""List the TF frames currently available (and key transforms): is the bridge publishing every sensor frame?

  python3 scripts/check_tf.py
"""
import _paths  # noqa: F401  (sets up sys.path)
import logging
import time

import rclpy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

log = logging.getLogger('check_tf')


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(name)s %(message)s')
    rclpy.init()
    node = rclpy.create_node('check_tf')
    buf = Buffer()
    TransformListener(buf, node)
    end = time.time() + 4.0
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    # all_frames_as_yaml(): one top-level "<frame>:" key per known frame
    frames = sorted(l.split(':')[0] for l in buf.all_frames_as_yaml().splitlines() if l and not l.startswith(' '))
    log.info('frames: %s', ', '.join(frames) or 'NONE (no /tf_static received)')
    for target, source in (('fsds/FSCar', 'fsds/Lidar1'), ('fsds/cam1', 'fsds/Lidar1')):
        try:
            t = buf.lookup_transform(target, source, Time()).transform.translation
            log.info('%s <- %s : (%.2f, %.2f, %.2f) m', target, source, t.x, t.y, t.z)
        except TransformException as e:
            log.warning('%s <- %s unavailable: %s', target, source, e)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
