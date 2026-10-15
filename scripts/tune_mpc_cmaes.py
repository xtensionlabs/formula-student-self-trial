#!/usr/bin/env python3
"""Tune the MPC cost weights with CMA-ES.

  python3 scripts/tune_mpc_cmaes.py --gens 15 --pop 12 --out models/mpc_weights.yaml
  python3 scripts/tune_mpc_cmaes.py --track recorded_track.npz      # tune on a real FSDS centerline

Each candidate = 7 weights searched in log-space. Fitness = mean `tracking_cost` over a fixed set of
closed-loop runs (random tracks by default, or a centerline recorded from FSDS: npz with `xy` (M,2)
and `v` (M,)). Writes a parameter snippet for fs_bringup/config/mpc.yaml.
"""
import _paths  # noqa: F401  (sets up sys.path)
import argparse
import logging
from dataclasses import fields
from multiprocessing import Pool
import cma
import numpy as np
from fs_control.mpc import Mpc, MpcWeights
from fs_control.reference import PathRef
from fs_control.sim import mpc_controller, random_track, run, tracking_cost

log = logging.getLogger('tune_mpc')
_mpc, _tracks, _T = None, None, 12.0


def _init(track_file: str, n_tracks: int, T: float) -> None:
    """Per-worker setup: build the (expensive) MPC once and fix the evaluation tracks."""
    global _mpc, _tracks, _T
    _mpc, _T = Mpc(), T
    if track_file:
        d = np.load(track_file)
        _tracks = [PathRef(d['xy'], d['v'])]
    else:
        _tracks = [random_track(np.random.default_rng(100 + i)) for i in range(n_tracks)]


def fitness(theta: np.ndarray) -> float:
    """Mean closed-loop cost of the weights exp(theta) over the evaluation tracks."""
    w = MpcWeights(*np.exp(theta))
    costs = [tracking_cost(run(mpc_controller(_mpc, w), t, T=_T, seed=i)) for i, t in enumerate(_tracks)]
    return float(np.mean(costs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gens', type=int, default=15)
    ap.add_argument('--pop', type=int, default=12)
    ap.add_argument('--tracks', type=int, default=3)
    ap.add_argument('--T', type=float, default=12.0, help='seconds per evaluation run')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--track', default='', help='npz with xy, v of a recorded FSDS centerline')
    ap.add_argument('--out', default='models/mpc_weights.yaml')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')

    x0 = np.log(MpcWeights().vec())
    es = cma.CMAEvolutionStrategy(x0, 0.5, {'popsize': a.pop, 'bounds': [-6, 7], 'seed': 1, 'verbose': -9})
    with Pool(a.workers, initializer=_init, initargs=(a.track, a.tracks, a.T)) as pool:
        base = pool.apply(fitness, (x0,))
        log.info('baseline cost %.4f', base)
        for g in range(a.gens):
            cand = es.ask()
            es.tell(cand, pool.map(fitness, cand))             # evaluate the population in parallel
            log.info('gen %2d  best %.4f  (baseline %.4f)', g, es.best.f, base)
    w = np.exp(es.best.x)
    with open(a.out, 'w') as f:
        f.write('mpc_controller_node:\n  ros__parameters:\n')
        for fld, v in zip(fields(MpcWeights), w):
            f.write(f'    w.{fld.name}: {v:.4f}\n')
    log.info('wrote %s (cost %.4f vs baseline %.4f)', a.out, es.best.f, base)


if __name__ == '__main__':
    main()
