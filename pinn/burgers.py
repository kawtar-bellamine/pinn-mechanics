"""Viscous Burgers equation, the classic PINN benchmark with a shock-like front.

u_t + u u_x = nu u_xx,  x in [-1, 1], t in [0, 1],  u(0, x) = -sin(pi x),  u(t, -1) = u(t, 1) = 0,  nu = 0.01/pi
Exact solution by the Cole-Hopf transform (Gauss-Hermite quadrature), a finite difference solver for comparison,
a forward PINN with and without residual-based adaptive refinement (RAR), and an inverse PINN that recovers nu
from noisy measurements.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .core import MLP, grad, train

NU = 0.01 / np.pi


def exact(t: np.ndarray, x: np.ndarray, nu: float = NU, n: int = 200) -> np.ndarray:
    """Cole-Hopf solution, evaluated in log space so that exp(-cos(pi y) / (2 pi nu)) does not overflow."""
    z, w = np.polynomial.hermite.hermgauss(n)
    t = np.asarray(t, float); x = np.asarray(x, float)
    out = np.empty(np.broadcast(t, x).shape)
    tt, xx = np.broadcast_arrays(t, x)
    for idx in np.ndindex(out.shape):
        ti, xi = tt[idx], xx[idx]
        if ti == 0:
            out[idx] = -np.sin(np.pi * xi); continue
        eta = np.sqrt(4 * nu * ti) * z
        y = xi - eta
        logf = -np.cos(np.pi * y) / (2 * np.pi * nu)
        m = logf.max()
        f = w * np.exp(logf - m)
        out[idx] = -np.sum(np.sin(np.pi * y) * f) / np.sum(f)
    return out


def fd_solve(nx: int = 2001, cfl: float = 0.4, nu: float = NU):
    """Explicit finite differences: conservative upwind (Rusanov flux) for u u_x, central for u_xx."""
    t0 = time.perf_counter()
    x = np.linspace(-1, 1, nx); dx = x[1] - x[0]
    u = -np.sin(np.pi * x)
    dt = min(cfl * dx / 1.0, 0.25 * dx * dx / nu)
    nt = int(np.ceil(1.0 / dt)); dt = 1.0 / nt
    snaps = {}
    want = {0.25: None, 0.5: None, 0.75: None, 1.0: None}
    for k in range(1, nt + 1):
        f = 0.5 * u ** 2
        a = np.maximum(np.abs(u[:-1]), np.abs(u[1:]))
        flux = 0.5 * (f[:-1] + f[1:]) - 0.5 * a * (u[1:] - u[:-1])
        un = u.copy()
        un[1:-1] = u[1:-1] - dt / dx * (flux[1:] - flux[:-1]) + nu * dt / dx ** 2 * (u[2:] - 2 * u[1:-1] + u[:-2])
        un[0] = un[-1] = 0.0
        u = un
        for tw in want:
            if abs(k * dt - tw) < dt / 2:
                snaps[tw] = u.copy()
    return x, snaps, time.perf_counter() - t0


def grid(nt=101, nx=256):
    t = np.linspace(0, 1, nt); x = np.linspace(-1, 1, nx)
    T, X = np.meshgrid(t, x, indexing="ij")
    return T, X


class Burgers:
    def __init__(self, width=40, depth=6, nu=None, seed=0):
        torch.manual_seed(seed)
        self.net = MLP(2, 1, width, depth, lo=[0, -1], hi=[1, 1])
        self.inverse = nu is None
        self.log_nu = torch.nn.Parameter(torch.tensor(np.log(0.05))) if self.inverse else None
        self.nu_fixed = nu

    def u(self, tx):
        # hard constraints: u(0,x) = -sin(pi x) and u(t, +-1) = 0 hold exactly
        t, x = tx[:, :1], tx[:, 1:]
        return -torch.sin(np.pi * x) + t * (1 - x ** 2) * self.net(tx)

    def nu(self):
        return torch.exp(self.log_nu) if self.inverse else self.nu_fixed

    def residual(self, tx):
        u = self.u(tx)
        g = grad(u, tx)
        u_t, u_x = g[:, :1], g[:, 1:]
        u_xx = grad(u_x, tx)[:, 1:]
        return u_t + u * u_x - self.nu() * u_xx

    def params(self):
        return list(self.net.parameters()) + ([self.log_nu] if self.inverse else [])

    def predict(self, T, X):
        with torch.no_grad():
            tx = torch.tensor(np.column_stack([T.ravel(), X.ravel()]))
            return self.u(tx).numpy().reshape(T.shape)


def fit_forward(n_col=4000, adam=3000, lbfgs=1500, rar_rounds=0, rar_add=300, seed=0):
    rng = np.random.default_rng(seed)
    model = Burgers(nu=NU, seed=seed)
    pts = np.column_stack([rng.uniform(0, 1, n_col), rng.uniform(-1, 1, n_col)])
    state = {"xc": torch.tensor(pts, requires_grad=True)}

    def loss_fn():
        r = model.residual(state["xc"])
        return (r ** 2).mean(), dict(pde=(r ** 2).mean())

    total, hist = 0.0, []
    h, s = train(model.params(), loss_fn, adam, lbfgs if rar_rounds == 0 else lbfgs // 2, lr=1e-3)
    hist += h; total += s
    for _ in range(rar_rounds):
        cand = np.column_stack([rng.uniform(0, 1, 20000), rng.uniform(-1, 1, 20000)])
        ct = torch.tensor(cand, requires_grad=True)
        r = model.residual(ct).detach().abs().numpy().ravel()
        top = cand[np.argsort(-r)[:rar_add]]
        state["xc"] = torch.tensor(np.vstack([state["xc"].detach().numpy(), top]), requires_grad=True)
        h, s = train(model.params(), loss_fn, 500, lbfgs // 2, lr=5e-4)
        hist += h; total += s
    return model, hist, total, state["xc"].detach().numpy()


def fit_inverse(n_data=2000, noise=0.01, n_col=3000, adam=2000, lbfgs=1000, seed=0):
    rng = np.random.default_rng(seed)
    td, xd = rng.uniform(0.05, 1, n_data), rng.uniform(-1, 1, n_data)
    ud = exact(td, xd)
    ud_noisy = ud + noise * np.abs(ud).max() * rng.standard_normal(n_data)
    model = Burgers(nu=None, seed=seed)
    xc = torch.tensor(np.column_stack([rng.uniform(0, 1, n_col), rng.uniform(-1, 1, n_col)]), requires_grad=True)
    txd = torch.tensor(np.column_stack([td, xd])); yd = torch.tensor(ud_noisy)[:, None]

    def loss_fn():
        r = model.residual(xc)
        d = model.u(txd) - yd
        parts = dict(pde=(r ** 2).mean(), data=(d ** 2).mean())
        return parts["pde"] + parts["data"], parts

    hist, secs = train(model.params(), loss_fn, adam, lbfgs, lr=1e-3)
    return model, hist, secs, float(model.nu().detach())
