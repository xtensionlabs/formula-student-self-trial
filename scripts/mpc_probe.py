#!/usr/bin/env python3
"""Run the MPC on the LIVE inputs (pose, path, speed profile) without sending any command to the car.

  python3 scripts/mpc_probe.py [--seconds 15]

Needs the stack running with control:=false. Prints solve time, IPOPT return status on failures, and the first
steering command, so solver trouble can be diagnosed while the car stands still.
"""
import _paths  # noqa: F401  (sets up sys.path)
import argparse
import logging
import time
from collections import Counter

import numpy as np
import rclpy

from fs_control.ctrl_common import ControlInputs, declare_common
from fs_control.mpc import Mpc, MpcWeights

log = logging.getLogger('mpc_probe')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=15.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(name)s %(message)s')
    rclpy.init()
    node = rclpy.create_node('mpc_probe')
    declare_common(node)
    inputs = ControlInputs(node)
    mpc, w, u_prev = Mpc(), MpcWeights(), np.zeros(2)
    stats, ms, shown = Counter(), [], 0
    end = time.time() + a.seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
        x, ref = inputs.state(), inputs.ref
        if x is None or ref is None:
            continue
        x0 = mpc.predict(x, u_prev)
        r = ref.horizon(x0[:2], x0[3], mpc.N, mpc.dt)
        u, info = mpc.solve(x0, r, u_prev, w)
        ms.append(info['ms'])
        stats['ok' if u is not None else info.get('status', 'failed')] += 1
        if u is not None:
            u_prev = u
        if shown < 6:
            shown += 1
            log.info('x0=%s  ref0=%s  refN=%s  -> %s', np.round(x0, 2), np.round(r[:, 0], 2), np.round(r[:, -1], 2),
                     'u=' + str(np.round(u, 3)) if u is not None else 'FAILED ' + info.get('status', ''))
        time.sleep(0.05)
    log.info('solves: %s   time ms: mean %.1f p95 %.1f max %.1f', dict(stats), np.mean(ms) if ms else np.nan,
             np.percentile(ms, 95) if ms else np.nan, np.max(ms) if ms else np.nan)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
