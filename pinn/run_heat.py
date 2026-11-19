"""Heat experiments: PINN (hard boundary conditions + refined sampling) and two ablations, against FEM."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .heat import evaluate, fem, fem_eval, fit

ROOT = Path(__file__).resolve().parents[1]


def main():
    res = {}
    b_ref, T_ref, s_ref = fem(128)
    X, Y = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 0.5, 101))
    ref = fem_eval(b_ref, T_ref, X, Y)
    res["fem"] = {}
    for n in (16, 32, 64):
        b, T, s = fem(n)
        res["fem"][f"mesh_{n}"] = dict(seconds=round(s, 3), dofs=int(b.N),
                                       rel_l2_vs_finest=float(np.linalg.norm(fem_eval(b, T, X, Y) - ref) /
                                                              np.linalg.norm(ref)))
    res["fem"]["mesh_128"] = dict(seconds=round(s_ref, 3), dofs=int(b_ref.N))
    res["reference_peak"] = float(ref.max())
    for name, kw in [("pinn_hard_bc_refined", dict(hard_bc=True, adaptive=True)),
                     ("pinn_hard_bc_uniform", dict(hard_bc=True, adaptive=False)),
                     ("pinn_soft_bc_refined", dict(hard_bc=False, adaptive=True))]:
        model, hist, secs = fit(**kw)
        m, *_ = evaluate(model, b_ref, T_ref)
        res[name] = dict(m, train_seconds=round(secs, 1), history=hist)
        torch.save(model.state_dict(), ROOT / f"results/{name}.pt")
        print(name, {k: v for k, v in m.items()}, round(secs), flush=True)
        (ROOT / "results/heat_metrics.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
