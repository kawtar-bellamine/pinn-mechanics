"""Burgers against the published benchmark (Raissi, Perdikaris and Karniadakis, J. Comput. Phys. 2019).

1. Our Cole-Hopf solution against the authors' reference data (burgers_shock.mat, 256 x 100 grid).
2. Our two trained PINNs (hard constraints, with and without adaptive refinement) on the same grid.
3. The paper's own set-up reproduced with our code: initial and boundary conditions as a loss term on N_u = 100
   points, N_f = 10,000 Latin hypercube collocation points, 8 hidden layers of 20 tanh units, L-BFGS only.
   The paper reports a relative L2 error of 6.7e-4 for this set-up.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch

from .burgers import NU, Burgers, exact
from .core import MLP, grad, train

ROOT = Path(__file__).resolve().parents[1]
PAPER_ERROR = 6.7e-4


def ref():
    d = scipy.io.loadmat(ROOT / "data/burgers_shock.mat")
    t, x, U = d["t"].ravel(), d["x"].ravel(), np.real(d["usol"])          # U: (256 x, 100 t)
    T, X = np.meshgrid(t, x)
    return T, X, U


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def raissi_setup(n_u=100, n_f=10_000, lbfgs=15_000, seed=1234):
    from scipy.stats import qmc
    T, X, U = ref()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    # initial condition and the two walls, as in the paper (N_u points drawn among 456 boundary points)
    xb = np.vstack([np.column_stack([T[:, 0], X[:, 0]]), np.column_stack([T[0, :], X[0, :]]),
                    np.column_stack([T[-1, :], X[-1, :]])])
    ub = np.concatenate([U[:, 0], U[0, :], U[-1, :]])
    pick = rng.choice(len(xb), n_u, replace=False)
    lhs = qmc.LatinHypercube(d=2, seed=seed).random(n_f)
    xf = np.vstack([np.column_stack([lhs[:, 0], -1 + 2 * lhs[:, 1]]), xb[pick]])
    net = MLP(2, 1, 20, 8, lo=[0, -1], hi=[1, 1])
    tb, yb = torch.tensor(xb[pick]), torch.tensor(ub[pick])[:, None]
    tf = torch.tensor(xf, requires_grad=True)

    def loss_fn():
        u = net(tf); g = grad(u, tf); u_t, u_x = g[:, :1], g[:, 1:]
        u_xx = grad(u_x, tf)[:, 1:]
        f = u_t + u * u_x - NU * u_xx
        parts = dict(data=((net(tb) - yb) ** 2).mean(), pde=(f ** 2).mean())
        return parts["data"] + parts["pde"], parts

    hist, secs = train(list(net.parameters()), loss_fn, adam_iters=0, lbfgs_iters=lbfgs)
    with torch.no_grad():
        P = net(torch.tensor(np.column_stack([T.ravel(), X.ravel()]))).numpy().reshape(T.shape)
    return P, secs, hist[-1]


def main():
    T, X, U = ref()
    out = {"reference": "Raissi et al. (2019), burgers_shock.mat, 256 x 100 grid"}
    E = exact(T.ravel(), X.ravel()).reshape(T.shape)
    out["cole_hopf_vs_published_reference"] = dict(rel_l2=rel(E, U), max_abs=float(np.abs(E - U).max()))
    for name in ("pinn", "pinn_rar"):
        f = ROOT / f"results/burgers_{name}.pt"
        if f.exists():
            m = Burgers(nu=NU); m.net.load_state_dict(torch.load(f))
            P = m.predict(T, X)
            out[f"our_{name}_on_published_grid"] = dict(rel_l2=rel(P, U), max_abs=float(np.abs(P - U).max()))
    t0 = time.time()
    P, secs, last = raissi_setup()
    out["paper_setup_reproduced"] = dict(rel_l2=rel(P, U), max_abs=float(np.abs(P - U).max()), train_seconds=round(secs),
                                         paper_reported_rel_l2=PAPER_ERROR, final_loss=last)
    np.save(ROOT / "results/burgers_paper_setup.npy", P)
    (ROOT / "results/reference_metrics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
