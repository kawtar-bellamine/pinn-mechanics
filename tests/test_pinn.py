import numpy as np
import torch

from pinn.burgers import Burgers, NU, exact, fd_solve
from pinn.core import MLP, grad
from pinn.heat import HeatPINN, fem, fem_eval


def test_cole_hopf_matches_initial_and_boundary_conditions():
    x = np.linspace(-1, 1, 11)
    assert np.allclose(exact(np.zeros_like(x), x), -np.sin(np.pi * x))
    assert abs(exact(np.array([0.5]), np.array([1.0]))[0]) < 1e-8


def test_finite_differences_close_to_exact():
    x, snaps, _ = fd_solve(nx=1001)
    ue = exact(np.full_like(x, 0.5), x)
    assert np.linalg.norm(snaps[0.5] - ue) / np.linalg.norm(ue) < 0.02


def test_fem_converges():
    X, Y = np.meshgrid(np.linspace(0, 1, 41), np.linspace(0, .5, 21))
    a = fem_eval(*fem(16)[:2], X, Y); b = fem_eval(*fem(48)[:2], X, Y)
    assert np.linalg.norm(a - b) / np.linalg.norm(b) < 1e-3
    assert np.allclose(b[:, 0], 0, atol=1e-4) and np.allclose(b[:, -1], 1, atol=1e-4)


def test_hard_constraints_hold_exactly():
    m = HeatPINN(hard_bc=True)
    y = torch.linspace(0, .5, 7)[:, None]
    with torch.no_grad():
        assert torch.allclose(m(torch.cat([torch.zeros_like(y), y], 1)), torch.zeros(7, 1))
        assert torch.allclose(m(torch.cat([torch.ones_like(y), y], 1)), torch.ones(7, 1))
    b = Burgers(nu=NU)
    x = torch.linspace(-1, 1, 9)[:, None]
    with torch.no_grad():
        assert torch.allclose(b.u(torch.cat([torch.zeros_like(x), x], 1)), -torch.sin(np.pi * x))


def test_autograd_second_derivative():
    x = torch.linspace(0, 1, 5, requires_grad=True)[:, None]
    u = x ** 3
    assert torch.allclose(grad(grad(u, x), x), 6 * x)


def test_trained_heat_model_matches_fem():
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "results/pinn_hard_bc_refined.pt"
    m = HeatPINN(); m.load_state_dict(torch.load(p))
    X, Y = np.meshgrid(np.linspace(0, 1, 41), np.linspace(0, .5, 21))
    ref = fem_eval(*fem(48)[:2], X, Y)
    with torch.no_grad():
        P = m(torch.tensor(np.column_stack([X.ravel(), Y.ravel()]))).numpy().reshape(X.shape)
    assert np.linalg.norm(P - ref) / np.linalg.norm(ref) < 0.01


def test_mlp_input_scaling():
    m = MLP(2, 1, lo=[0, -1], hi=[1, 1])
    assert m(torch.zeros(3, 2)).shape == (3, 1)


def test_cole_hopf_matches_the_published_reference():
    from pinn.navier_stokes import fetch
    from pinn.reference import ref, rel
    fetch()
    T, X, U = ref()
    assert rel(exact(T.ravel(), X.ravel()).reshape(T.shape), U) < 1e-9


def test_stream_function_gives_divergence_free_flow():
    import torch
    from pinn.core import grad
    from pinn.navier_stokes import NS
    m = NS([1, -2, 0], [8, 2, 20])
    xyt = torch.rand(64, 3, requires_grad=True) * torch.tensor([7.0, 4.0, 20.0]) + torch.tensor([1.0, -2.0, 0.0])
    xyt = xyt.detach().requires_grad_(True)
    u, v, _ = m.fields(xyt)
    div = grad(u, xyt)[:, :1] + grad(v, xyt)[:, 1:2]
    assert float(div.abs().max()) < 1e-10
