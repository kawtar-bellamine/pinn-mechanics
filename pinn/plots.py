"""Figures for the README."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .burgers import exact, fd_solve, grid  # noqa: E402
from .heat import HeatPINN, LX, LY, fem, fem_eval  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures"


def heat():
    r = json.loads((ROOT / "results/heat_metrics.json").read_text())
    b, T, _ = fem(128)
    X, Y = np.meshgrid(np.linspace(0, LX, 201), np.linspace(0, LY, 101))
    ref = fem_eval(b, T, X, Y)
    preds = {}
    for name in ("pinn_hard_bc_refined", "pinn_hard_bc_uniform", "pinn_soft_bc_refined"):
        m = HeatPINN(hard_bc="hard" in name)
        m.load_state_dict(torch.load(ROOT / f"results/{name}.pt"))
        with torch.no_grad():
            preds[name] = m(torch.tensor(np.column_stack([X.ravel(), Y.ravel()]))).numpy().reshape(X.shape)
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), dpi=130)
    vmax = ref.max()
    panels = [(ref, "Finite elements (P2, 131,841 dof)", "inferno", (0, vmax)),
              (preds["pinn_hard_bc_refined"], f"PINN, exact boundary conditions + refined sampling\n"
               f"error {100 * r['pinn_hard_bc_refined']['rel_l2']:.2f}%", "inferno", (0, vmax)),
              (np.abs(preds["pinn_hard_bc_refined"] - ref), "|PINN - FEM| (max "
               f"{r['pinn_hard_bc_refined']['max_abs']:.4f})", "viridis", None),
              (preds["pinn_hard_bc_uniform"], f"PINN, uniform collocation: training stuck\nerror "
               f"{100 * r['pinn_hard_bc_uniform']['rel_l2']:.0f}%", "inferno", (0, vmax))]
    for ax, (Z, t, cm, lim) in zip(axes.ravel(), panels):
        im = ax.imshow(Z, origin="lower", extent=[0, LX, 0, LY], cmap=cm,
                       vmin=lim[0] if lim else None, vmax=lim[1] if lim else None, aspect="equal")
        ax.set_title(t, fontsize=9.5); fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle("Steady heat conduction with a concentrated source", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig1_heat.png"); plt.close(fig)


def burgers():
    r = json.loads((ROOT / "results/burgers_metrics.json").read_text())
    T, X = grid(101, 256)
    U = np.load(ROOT / "results/burgers_exact.npy")
    P = np.load(ROOT / "results/burgers_pinn.npy"); PR = np.load(ROOT / "results/burgers_pinn_rar.npy")
    pts = np.load(ROOT / "results/burgers_pinn_rar_points.npy")
    fig = plt.figure(figsize=(12, 6.4), dpi=130)
    ax = fig.add_subplot(2, 1, 1)
    im = ax.imshow(U.T, origin="lower", extent=[0, 1, -1, 1], aspect="auto", cmap="RdBu_r")
    n0 = r["pinn_rar"]["collocation_points"] - 900
    ax.scatter(pts[n0:, 0], pts[n0:, 1], s=2, c="k", alpha=.5, label="points added by adaptive refinement")
    ax.set_xlabel("t"); ax.set_ylabel("x"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Exact solution u(t, x) (Cole-Hopf); the refinement puts points on the steep front", fontsize=10)
    fig.colorbar(im, ax=ax)
    xf, snaps, _ = fd_solve()
    for i, t in enumerate((0.25, 0.5, 0.75)):
        a = fig.add_subplot(2, 3, 4 + i)
        k = int(t * 100)
        a.plot(X[k], U[k], "k-", lw=2.2, label="exact")
        a.plot(X[k], P[k], "--", color="#e69138", lw=1.4, label="PINN")
        a.plot(X[k], PR[k], ":", color="#1f5fa8", lw=1.8, label="PINN + refinement")
        a.set_title(f"t = {t}", fontsize=10); a.set_xlabel("x")
        if i == 0:
            a.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig2_burgers.png"); plt.close(fig)


def inverse():
    r = json.loads((ROOT / "results/burgers_metrics.json").read_text())["inverse"]
    labels = [k.replace("_points_", " points\n").replace("pct_noise", "% noise") for k in r]
    vals = [v["rel_error_pct"] for v in r.values()]
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=130)
    b = ax.bar(labels, vals, color="#1f5fa8")
    ax.bar_label(b, labels=[f"{v:.1f}%" for v in vals], fontsize=9)
    ax.set_ylabel("error on the identified viscosity")
    ax.set_title("Inverse problem: viscosity recovered from noisy measurements", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig3_inverse.png"); plt.close(fig)


def navier_stokes():
    f = ROOT / "results/ns_fields.npz"
    if not f.exists():
        return
    d = np.load(f); m = json.loads((ROOT / "results/ns_metrics.json").read_text())[0]
    key = "noise_0.0__"
    x, y = d[key + "x"], d[key + "y"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), dpi=140)
    for row, (q, lab, cm) in enumerate((("u", "streamwise velocity u", "RdBu_r"), ("p", "pressure (mean removed)",
                                                                                    "viridis"))):
        t, pr = d[key + q + "_true"], d[key + q + "_pred"]
        lim = (min(t.min(), pr.min()), max(t.max(), pr.max()))
        for col, (v, src) in enumerate(((t, "DNS (Nektar++, Raissi et al. 2019)"), (pr, "PINN"))):
            ax = axes[row, col]
            tc = ax.tricontourf(x, y, v, levels=40, cmap=cm, vmin=lim[0], vmax=lim[1])
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{lab}: {src}", fontsize=9.5)
        fig.colorbar(tc, ax=axes[row, :].tolist(), shrink=0.85, pad=0.01)
    fig.suptitle(f"Cylinder wake, Re = 100, t = 10 s. Pressure never observed: error {m['err_p_pct']:.1f}%; "
                 f"identified 1/Re = {m['lambda2']:.5f} (true 0.01)", fontsize=10.5)
    fig.savefig(OUT / "fig4_navier_stokes.png", bbox_inches="tight"); plt.close(fig)


def main():
    navier_stokes()
    heat(); burgers(); inverse()


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:] or ["heat", "burgers", "inverse", "navier_stokes"]:
        globals()[f]()
