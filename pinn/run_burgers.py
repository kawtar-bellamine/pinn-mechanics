"""Burgers experiments: finite differences, PINN, PINN + adaptive refinement, inverse PINN (recover nu)."""
import json
from pathlib import Path

import numpy as np
import torch

from .burgers import NU, exact, fd_solve, fit_forward, fit_inverse, grid

ROOT = Path(__file__).resolve().parents[1]


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    T, X = grid(101, 256)
    U = exact(T, X)
    np.save(ROOT / "results/burgers_exact.npy", U)
    res = {}
    x, snaps, sec = fd_solve()
    res["finite_differences"] = dict(seconds=round(sec, 3), nx=len(x),
                                     rel_l2_at_t={str(t): rel(u, exact(np.full_like(x, t), x)) for t, u in snaps.items()})
    for name, kw in [("pinn", dict(rar_rounds=0)), ("pinn_rar", dict(rar_rounds=3))]:
        model, hist, secs, pts = fit_forward(**kw)
        P = model.predict(T, X)
        np.save(ROOT / f"results/burgers_{name}.npy", P)
        np.save(ROOT / f"results/burgers_{name}_points.npy", pts)
        res[name] = dict(rel_l2=rel(P, U), max_abs=float(np.abs(P - U).max()), train_seconds=round(secs, 1),
                         collocation_points=len(pts), history=hist,
                         rel_l2_at_t={str(t): rel(P[int(t * 100)], U[int(t * 100)]) for t in (0.25, 0.5, 0.75, 1.0)})
        torch.save(model.net.state_dict(), ROOT / f"results/burgers_{name}.pt")
        print(name, res[name]["rel_l2"], res[name]["max_abs"], secs, flush=True)
        (ROOT / "results/burgers_metrics.json").write_text(json.dumps(res, indent=1))
    res["inverse"] = {}
    for n_data, noise in [(2000, 0.01), (500, 0.01), (200, 0.05)]:
        model, hist, secs, nu = fit_inverse(n_data=n_data, noise=noise)
        res["inverse"][f"{n_data}_points_{int(noise * 100)}pct_noise"] = dict(
            nu_true=NU, nu_found=nu, rel_error_pct=100 * abs(nu - NU) / NU, train_seconds=round(secs, 1),
            field_rel_l2=rel(model.predict(T, X), U))
        print("inverse", n_data, noise, nu, NU, flush=True)
        (ROOT / "results/burgers_metrics.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
