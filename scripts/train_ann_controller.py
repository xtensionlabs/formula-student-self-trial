#!/usr/bin/env python3
"""Train the FF-ANN controller by imitating the MPC on simulated tracks.

  python3 scripts/train_ann_controller.py --tracks 60 --epochs 80 --out models/ann_controller.pt

Pipeline: MPC drives random tracks (noise added to the executed command so the data also covers
off-nominal states; the label is always the clean MPC command) -> (features, [delta, a]) pairs ->
MLP -> TorchScript file loaded by ann_controller_node.
"""
import _paths  # noqa: F401  (sets up sys.path)
import argparse
import logging
from multiprocessing import Pool
import numpy as np
import torch
from torch import nn
from fs_control.mpc import Mpc, MpcWeights
from fs_control.sim import mpc_controller, random_track, run, tracking_cost

log = logging.getLogger('train_ann')
D_MAX, A_SCALE = 0.436, 10.0
FEAT_SCALE = [1.0, 0.5, 5.0, 0.2]            # rough magnitudes of [cte, heading_err, speed_err, curvature]
_mpc = None


class AnnPolicy(nn.Module):
    """features (B,4) -> [delta, a] (B,2); feature scaling and output limits are inside the model."""

    def __init__(self) -> None:
        super().__init__()
        self.d_max = D_MAX                                       # attribute: usable inside TorchScript
        self.register_buffer('scale', torch.tensor(FEAT_SCALE))
        self.net = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        o = self.net(f / self.scale)
        delta = self.d_max * torch.tanh(o[:, 0:1])                 # |delta| <= steering limit
        a = -10.0 + 15.0 * torch.sigmoid(o[:, 1:2])           # a in [-10, 5]
        return torch.cat([delta, a], dim=1)


def _collect(seed: int) -> np.ndarray:
    """One MPC drive on a random track; returns rows [features(4), delta, a]."""
    global _mpc
    _mpc = _mpc or Mpc()
    rng = np.random.default_rng(seed)
    rec: list = []
    noise = (0.04, 0.8) if seed % 2 else (0.0, 0.0)           # half of the runs explore with noise
    run(mpc_controller(_mpc, MpcWeights(), rec, noise, seed), random_track(rng), T=25, seed=seed,
        offset=rng.uniform(-1.0, 1.0), v0=rng.uniform(2, 8))
    return np.array(rec).reshape(-1, 6)


def _evaluate(model: nn.Module, seeds) -> None:
    """Closed-loop comparison ANN vs MPC on held-out tracks."""
    global _mpc
    _mpc = _mpc or Mpc()

    def ann(x, ref, u_prev):
        with torch.no_grad():
            return model(torch.tensor(ref.features(x[:3], x[3]), dtype=torch.float32)[None])[0].numpy().astype(float)

    for name, ctrl in (('MPC', lambda s: mpc_controller(_mpc, MpcWeights())), ('ANN', lambda s: ann)):
        res = [run(ctrl(s), random_track(np.random.default_rng(s)), T=20, seed=s) for s in seeds]
        rms = np.mean([np.sqrt(np.mean(r['feat'][:, 0] ** 2)) for r in res])
        log.info('%s: rms cte %.3f m, off-track %d/%d, mean cost %.2f', name, rms,
                 sum(r['off_track'] for r in res), len(res), np.mean([tracking_cost(r) for r in res]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', type=int, default=60)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default='models/ann_controller.pt')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')

    with Pool(a.workers) as pool:                              # parallel MPC data collection
        data = np.vstack(pool.map(_collect, range(a.tracks)))
    log.info('collected %d samples from %d tracks', len(data), a.tracks)

    X = torch.tensor(data[:, :4], dtype=torch.float32)
    Y = torch.tensor(data[:, 4:], dtype=torch.float32)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
    n_val = len(X) // 10
    val, tr = perm[:n_val], perm[n_val:]
    model = AnnPolicy()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    w = torch.tensor([1 / D_MAX, 1 / A_SCALE])                 # balance steering and accel errors
    for ep in range(a.epochs):
        idx = tr[torch.randperm(len(tr))]
        for b in range(0, len(idx), 256):
            j = idx[b:b + 256]
            loss = (((model(X[j]) - Y[j]) * w) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == a.epochs - 1:
            with torch.no_grad():
                log.info('epoch %3d  val loss %.5f', ep, (((model(X[val]) - Y[val]) * w) ** 2).mean().item())
    model.eval()
    torch.jit.script(model).save(a.out)                        # TorchScript: loaded without this file
    log.info('saved %s', a.out)
    _evaluate(model, range(1000, 1005))


if __name__ == '__main__':
    main()
