"""Shared PINN building blocks: network, derivatives, training loop (Adam then L-BFGS)."""
from __future__ import annotations

import time

import torch

torch.set_default_dtype(torch.float64)


class MLP(torch.nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int = 50, depth: int = 4, lo=None, hi=None):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
            d = width
        layers.append(torch.nn.Linear(d, n_out))
        self.net = torch.nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight); torch.nn.init.zeros_(m.bias)
        self.register_buffer("lo", torch.tensor(lo if lo is not None else [0.0] * n_in))
        self.register_buffer("hi", torch.tensor(hi if hi is not None else [1.0] * n_in))

    def forward(self, x):
        return self.net(2 * (x - self.lo) / (self.hi - self.lo) - 1)


def grad(u, x):
    return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]


def train(params, loss_fn, adam_iters=4000, lbfgs_iters=500, lr=1e-3, log_every=500, callback=None):
    """Adam for the rough fit, then L-BFGS to converge: the usual recipe for PINNs."""
    hist, t0 = [], time.perf_counter()
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(adam_iters // 3, 1), gamma=0.5)
    for it in range(adam_iters):
        opt.zero_grad()
        loss, parts = loss_fn()
        loss.backward(); opt.step(); sched.step()
        if it % log_every == 0:
            hist.append(dict(it=it, loss=float(loss.detach()), **{k: float(v.detach()) for k, v in parts.items()}))
        if callback and it and it % log_every == 0:
            callback(it)
    if lbfgs_iters:
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=lbfgs_iters, history_size=50,
                                tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss, _ = loss_fn()
            loss.backward()
            return loss
        opt.step(closure)
        loss, parts = loss_fn()
        hist.append(dict(it=adam_iters + lbfgs_iters, loss=float(loss.detach()),
                         **{k: float(v.detach()) for k, v in parts.items()}))
    return hist, time.perf_counter() - t0
