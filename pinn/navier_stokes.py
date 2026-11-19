"""Inverse Navier-Stokes problem on real simulation data: the flow past a circular cylinder at Re = 100.

Data: the 2D incompressible wake computed by direct numerical simulation (spectral/hp elements, Nektar++), published
with Raissi, Perdikaris and Karniadakis, "Physics-informed neural networks", J. Comput. Phys. 378 (2019),
repository maziarraissi/PINNs (file main/Data/cylinder_nektar_wake.mat): 5,000 points in the wake x 200 time steps,
velocity and pressure.

Problem (as in the paper): from velocity samples only (u, v at N random points of space-time, no pressure at all),
learn the flow and the two unknown coefficients of

    u_t + l1 (u u_x + v u_y) = -p_x + l2 (u_xx + u_yy)
    v_t + l1 (u v_x + v v_y) = -p_y + l2 (v_xx + v_yy)

True values: l1 = 1, l2 = 1/Re = 0.01. The network outputs a stream function psi and the pressure p, with
u = psi_y and v = -psi_x, so the flow is divergence-free by construction. The pressure is never observed: it is
identified up to a constant, and compared with the simulation's after removing the mean.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch

from .core import MLP, grad, train

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/cylinder_nektar_wake.mat"
URL_REPO = "https://github.com/maziarraissi/PINNs.git"
COMMIT = "932f50a2d8ef4e80d1456bbae6887a73ff5166ef"


def fetch():
    """Sparse clone of the authors' repository at a pinned commit, then copy the .mat files used here."""
    import shutil
    import subprocess
    import tempfile
    if DATA.exists() and (ROOT / "data/burgers_shock.mat").exists():
        return
    DATA.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--sparse", "--no-checkout", URL_REPO, d],
                       check=True)
        subprocess.run(["git", "-C", d, "sparse-checkout", "set", "main/Data", "appendix/Data"], check=True)
        subprocess.run(["git", "-C", d, "checkout", "-q", COMMIT], check=True)
        shutil.copy(Path(d) / "main/Data/cylinder_nektar_wake.mat", DATA)
        shutil.copy(Path(d) / "appendix/Data/burgers_shock.mat", ROOT / "data/burgers_shock.mat")


def load():
    d = scipy.io.loadmat(DATA)
    X, t = d["X_star"], d["t"].ravel()          # (5000, 2), (200,)
    U, P = d["U_star"], d["p_star"]             # (5000, 2, 200), (5000, 200)
    return X, t, U, P


class NS:
    def __init__(self, lo, hi, width=20, depth=8, seed=0):
        torch.manual_seed(seed)
        self.net = MLP(3, 2, width, depth, lo=lo, hi=hi)
        self.l1 = torch.nn.Parameter(torch.tensor(0.0))
        self.l2 = torch.nn.Parameter(torch.tensor(0.0))

    def fields(self, xyt):
        out = self.net(xyt)
        psi, p = out[:, :1], out[:, 1:]
        g = grad(psi, xyt)
        u, v = g[:, 1:2], -g[:, :1]
        return u, v, p

    def residual(self, xyt):
        u, v, p = self.fields(xyt)
        gu, gv, gp = grad(u, xyt), grad(v, xyt), grad(p, xyt)
        u_x, u_y, u_t = gu[:, :1], gu[:, 1:2], gu[:, 2:]
        v_x, v_y, v_t = gv[:, :1], gv[:, 1:2], gv[:, 2:]
        u_xx, u_yy = grad(u_x, xyt)[:, :1], grad(u_y, xyt)[:, 1:2]
        v_xx, v_yy = grad(v_x, xyt)[:, :1], grad(v_y, xyt)[:, 1:2]
        f = u_t + self.l1 * (u * u_x + v * u_y) + gp[:, :1] - self.l2 * (u_xx + u_yy)
        g_ = v_t + self.l1 * (u * v_x + v * v_y) + gp[:, 1:2] - self.l2 * (v_xx + v_yy)
        return u, v, p, f, g_

    def params(self):
        return list(self.net.parameters()) + [self.l1, self.l2]


def fit(n_train=5000, noise=0.0, adam=20000, lbfgs=3000, batch=1000, seed=0):
    """Adam on random mini-batches of the data points (cheap steps), then L-BFGS on all of them. Single precision:
    this is the CPU budget; the paper trained far longer on a GPU."""
    torch.set_default_dtype(torch.float32)
    try:
        return _fit(n_train, noise, adam, lbfgs, batch, seed)
    finally:
        torch.set_default_dtype(torch.float64)


def _fit(n_train, noise, adam, lbfgs, batch, seed):
    X, t, U, P = load()
    N, T = X.shape[0], t.size
    rng = np.random.default_rng(seed)
    idx = rng.choice(N * T, n_train, replace=False)
    i, k = idx // T, idx % T
    xyt = np.column_stack([X[i, 0], X[i, 1], t[k]])
    uv = np.column_stack([U[i, 0, k], U[i, 1, k]])
    if noise:
        uv = uv + noise * uv.std(0) * rng.standard_normal(uv.shape)
    lo, hi = [float(v) for v in xyt.min(0)], [float(v) for v in xyt.max(0)]
    model = NS(lo, hi, seed=seed)
    xt_all = torch.tensor(xyt, dtype=torch.float32); yt_all = torch.tensor(uv, dtype=torch.float32)
    state = {"calls": 0}
    gen = torch.Generator().manual_seed(seed)

    def loss_fn():
        state["calls"] += 1
        if state["calls"] <= adam and batch < n_train:
            b = torch.randint(0, n_train, (batch,), generator=gen)
            xt, yt = xt_all[b].clone().requires_grad_(True), yt_all[b]
        else:
            xt, yt = xt_all.clone().requires_grad_(True), yt_all
        u, v, _, f, g_ = model.residual(xt)
        parts = dict(data=((u - yt[:, :1]) ** 2).mean() + ((v - yt[:, 1:]) ** 2).mean(),
                     pde=(f ** 2).mean() + (g_ ** 2).mean())
        return parts["data"] + parts["pde"], parts

    hist, secs = train(model.params(), loss_fn, adam, lbfgs, lr=1e-3, log_every=1000)
    # evaluation on a full snapshot never sampled as such (t index 100, as in the paper)
    k0 = 100
    xyt0 = torch.tensor(np.column_stack([X[:, 0], X[:, 1], np.full(N, t[k0])]), dtype=torch.float32,
                        requires_grad=True)
    u, v, p = model.fields(xyt0)
    u, v, p = (z.detach().numpy().ravel() for z in (u, v, p))
    pe = P[:, k0] - P[:, k0].mean(); pp = p - p.mean()
    l1, l2 = float(model.l1.detach()), float(model.l2.detach())
    res = dict(n_train=n_train, noise=noise, seconds=round(secs, 1), lambda1=l1, lambda2=l2,
               lambda1_err_pct=round(100 * abs(l1 - 1.0), 3), lambda2_err_pct=round(100 * abs(l2 - 0.01) / 0.01, 2),
               err_u_pct=round(100 * np.linalg.norm(u - U[:, 0, k0]) / np.linalg.norm(U[:, 0, k0]), 3),
               err_v_pct=round(100 * np.linalg.norm(v - U[:, 1, k0]) / np.linalg.norm(U[:, 1, k0]), 3),
               err_p_pct=round(100 * np.linalg.norm(pp - pe) / np.linalg.norm(pe), 2),
               final_loss=hist[-1])
    field = dict(x=X[:, 0], y=X[:, 1], u_true=U[:, 0, k0], u_pred=u, p_true=pe, p_pred=pp)
    return model, res, field


def _run(noise):
    torch.set_num_threads(1)
    model, res, field = fit(noise=noise, adam=12000, lbfgs=2000)
    return res, field


def main():
    from multiprocessing import Pool
    fetch()
    with Pool(2) as pool:
        runs = pool.map(_run, [0.0, 0.01])
    out = [r for r, _ in runs]
    print(json.dumps(out, indent=1, default=float), flush=True)
    (ROOT / "results/ns_metrics.json").write_text(json.dumps(out, indent=1, default=float))
    np.savez(ROOT / "results/ns_fields.npz", **{f"noise_{r['noise']}__{k}": v for r, f in runs for k, v in f.items()})


if __name__ == "__main__":
    main()
