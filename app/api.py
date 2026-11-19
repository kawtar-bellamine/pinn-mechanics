"""Serve the trained PINNs: any point of the field on request, no mesh, no solver call."""
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, Query

from pinn.burgers import Burgers, NU
from pinn.heat import HeatPINN, LX, LY

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="PINN surrogates: heat conduction and Burgers flow")

heat = HeatPINN(hard_bc=True)
heat.load_state_dict(torch.load(ROOT / "results/pinn_hard_bc_refined.pt"))
burg = Burgers(nu=NU)
burg.net.load_state_dict(torch.load(ROOT / "results/burgers_pinn_rar.pt"))


@app.get("/heat/field")
def heat_field(nx: int = Query(101, le=1001), ny: int = Query(51, le=501)):
    X, Y = np.meshgrid(np.linspace(0, LX, nx), np.linspace(0, LY, ny))
    with torch.no_grad():
        T = heat(torch.tensor(np.column_stack([X.ravel(), Y.ravel()]))).numpy().reshape(X.shape)
    return {"x": X[0].tolist(), "y": Y[:, 0].tolist(), "T": T.round(5).tolist(), "T_max": float(T.max())}


@app.get("/heat/point")
def heat_point(x: float = Query(..., ge=0, le=LX), y: float = Query(..., ge=0, le=LY)):
    with torch.no_grad():
        return {"x": x, "y": y, "T": float(heat(torch.tensor([[x, y]]))[0, 0])}


@app.get("/burgers/profile")
def burgers_profile(t: float = Query(..., ge=0, le=1), nx: int = Query(201, le=2001)):
    x = np.linspace(-1, 1, nx)
    u = burg.predict(np.full_like(x, t), x)
    return {"t": t, "x": x.tolist(), "u": u.round(5).tolist()}
