"""Steady heat conduction in a plate with a concentrated source: PINN against a finite element reference.

-k (T_xx + T_yy) = q(x, y)  on [0, 1] x [0, 0.5]
T = 0 on the left edge, T = 1 on the right edge, insulated top and bottom edges (dT/dn = 0).
q is a Gaussian heat source (sigma = 0.05) centred at (0.3, 0.25): a sharp feature the network has to resolve.
Reference: P2 finite elements (scikit-fem) on a fine mesh, checked by refinement.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from skfem import Basis, BilinearForm, ElementTriP2, LinearForm, MeshTri, condense, solve
from skfem.helpers import dot, grad as fgrad

from .core import MLP, grad, train

LX, LY, K = 1.0, 0.5, 1.0
X0, Y0, SIG, Q0 = 0.3, 0.25, 0.05, 500.0


def source(x, y, lib=np):
    return Q0 * lib.exp(-((x - X0) ** 2 + (y - Y0) ** 2) / (2 * SIG ** 2))


# ---------- finite elements ----------
def fem(n: int = 64):
    t0 = time.perf_counter()
    m = MeshTri.init_tensor(np.linspace(0, LX, 2 * n + 1), np.linspace(0, LY, n + 1))
    basis = Basis(m, ElementTriP2())

    @BilinearForm
    def a(u, v, w):
        return K * dot(fgrad(u), fgrad(v))

    @LinearForm
    def L(v, w):
        return source(w.x[0], w.x[1]) * v

    A, b = a.assemble(basis), L.assemble(basis)
    left = basis.get_dofs(lambda x: np.isclose(x[0], 0))
    right = basis.get_dofs(lambda x: np.isclose(x[0], LX))
    T = np.zeros(basis.N)
    T[right] = 1.0
    D = np.concatenate([left.all(), right.all()])
    T = solve(*condense(A, b, x=T, D=D))
    return basis, T, time.perf_counter() - t0


def fem_eval(basis, T, X, Y):
    """Interpolate the P2 solution from its nodes (vertices and edge midpoints) to arbitrary points."""
    from scipy.interpolate import CloughTocher2DInterpolator
    f = CloughTocher2DInterpolator(basis.doflocs.T, T)
    return f(X, Y)


# ---------- PINN ----------
class HeatPINN(torch.nn.Module):
    def __init__(self, hard_bc: bool = True, width=50, depth=4):
        super().__init__()
        self.net = MLP(2, 1, width, depth, lo=[0, 0], hi=[LX, LY])
        self.hard = hard_bc

    def forward(self, xy):
        n = self.net(xy)
        if self.hard:      # T = x/LX + x (LX - x) N : the Dirichlet conditions hold exactly
            x = xy[:, :1]
            return x / LX + x * (LX - x) * n
        return n


def fit(hard_bc=True, n_col=3000, adam=2500, lbfgs=1000, seed=0, adaptive=True):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = HeatPINN(hard_bc)
    # collocation: uniform + extra points around the source (where the curvature is)
    pts = np.column_stack([rng.uniform(0, LX, n_col), rng.uniform(0, LY, n_col)])
    if adaptive:
        extra = np.column_stack([rng.normal(X0, 2 * SIG, n_col // 3), rng.normal(Y0, 2 * SIG, n_col // 3)])
        extra = extra[(extra[:, 0] > 0) & (extra[:, 0] < LX) & (extra[:, 1] > 0) & (extra[:, 1] < LY)]
        pts = np.vstack([pts, extra])
    xc = torch.tensor(pts, requires_grad=True)
    nb = 400
    xb = torch.tensor(np.column_stack([rng.uniform(0, LX, nb), np.r_[np.zeros(nb // 2), np.full(nb // 2, LY)]]),
                      requires_grad=True)
    yl = torch.tensor(np.column_stack([np.zeros(nb // 2), rng.uniform(0, LY, nb // 2)]))
    yr = torch.tensor(np.column_stack([np.full(nb // 2, LX), rng.uniform(0, LY, nb // 2)]))
    q = torch.tensor(source(pts[:, 0], pts[:, 1]))[:, None]

    def loss_fn():
        T = model(xc)
        g = grad(T, xc)
        lap = grad(g[:, :1], xc)[:, :1] + grad(g[:, 1:], xc)[:, 1:]
        r = (K * lap + q) / Q0
        Tb = model(xb)
        neu = grad(Tb, xb)[:, 1:]
        parts = dict(pde=(r ** 2).mean(), neumann=(neu ** 2).mean())
        loss = parts["pde"] + parts["neumann"]
        if not hard_bc:
            parts["dirichlet"] = (model(yl) ** 2).mean() + ((model(yr) - 1) ** 2).mean()
            loss = loss + 10 * parts["dirichlet"]
        return loss, parts

    hist, secs = train(list(model.parameters()), loss_fn, adam, lbfgs, lr=2e-3)
    return model, hist, secs


def evaluate(model, basis, T, nx=201, ny=101):
    X, Y = np.meshgrid(np.linspace(0, LX, nx), np.linspace(0, LY, ny))
    ref = fem_eval(basis, T, X, Y)
    with torch.no_grad():
        t0 = time.perf_counter()
        pred = model(torch.tensor(np.column_stack([X.ravel(), Y.ravel()]))).numpy().reshape(X.shape)
        inf = time.perf_counter() - t0
    rel = float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))
    return dict(rel_l2=rel, max_abs=float(np.abs(pred - ref).max()), peak_ref=float(ref.max()),
                peak_pinn=float(pred.max()), inference_s=inf), X, Y, ref, pred
