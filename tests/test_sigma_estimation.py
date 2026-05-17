"""Tests for torchcfm.sigma_estimation.

Key test: estimate_sigma_bounds on a known linear map T(x) = Ax
where A is diagonal, so σ*_max = max(diag(A)) and σ*_min = min(diag(A)).
"""

import torch

from torchcfm.sigma_estimation import estimate_sigma_bounds


def _linear_v_fn(A: torch.Tensor):
    """Velocity field for the displacement interpolation of T(x) = Ax.

    The path is X(t; x0) = (1-t)x0 + t*A@x0 = (I + t(A-I)) x0.
    The velocity is:
        dX/dt = (A - I) x0
    Expressing x0 = (I + t(A-I))^{-1} X:
        v(X, t) = (A - I) (I + t(A-I))^{-1} X

    For diagonal A = diag(a_i) this simplifies to:
        v_i(X, t) = (a_i - 1) / (1 + t(a_i - 1)) * X_i
    """
    def v_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (bs, d), t: (bs,)
        t_col = t.view(-1, 1)
        diag = A.to(x.device, x.dtype)
        denom = 1.0 + t_col * (diag - 1.0)  # (bs, d)
        return (diag - 1.0) / denom * x

    return v_fn


def test_sigma_bounds_diagonal_linear_autodiff():
    """Diagonal linear map: estimated bounds should be close to exact eigenvalues."""
    torch.manual_seed(42)
    a1, a2 = 2.0, 0.5
    A = torch.tensor([a1, a2])
    v_fn = _linear_v_fn(A)

    sigma_min_star, sigma_max_star = estimate_sigma_bounds(
        v_fn=v_fn,
        dim=2,
        n_samples=64,
        device=torch.device("cpu"),
        n_ode_steps=200,
        q_low=0.05,
        q_high=0.95,
        jacobian_method="autodiff",
    )

    # Allow 10% relative tolerance — quantile estimation and ODE error
    assert abs(sigma_max_star - a1) / a1 < 0.10, (
        f"σ*_max={sigma_max_star:.4f} far from expected {a1}"
    )
    assert abs(sigma_min_star - a2) / a2 < 0.10, (
        f"σ*_min={sigma_min_star:.4f} far from expected {a2}"
    )


def test_sigma_bounds_diagonal_linear_variational():
    """Same test via variational ODE path."""
    torch.manual_seed(42)
    a1, a2 = 2.0, 0.5
    A = torch.tensor([a1, a2])
    v_fn = _linear_v_fn(A)

    sigma_min_star, sigma_max_star = estimate_sigma_bounds(
        v_fn=v_fn,
        dim=2,
        n_samples=64,
        device=torch.device("cpu"),
        n_ode_steps=200,
        q_low=0.05,
        q_high=0.95,
        jacobian_method="variational",
    )

    assert abs(sigma_max_star - a1) / a1 < 0.10, (
        f"σ*_max={sigma_max_star:.4f} far from expected {a1}"
    )
    assert abs(sigma_min_star - a2) / a2 < 0.10, (
        f"σ*_min={sigma_min_star:.4f} far from expected {a2}"
    )


def test_sigma_bounds_autodiff_variational_agree():
    """Both methods should return close results on the same problem."""
    torch.manual_seed(0)
    A = torch.tensor([1.5, 0.7])
    v_fn = _linear_v_fn(A)

    kwargs = dict(v_fn=v_fn, dim=2, n_samples=32, n_ode_steps=100)
    s_min_ad, s_max_ad = estimate_sigma_bounds(**kwargs, jacobian_method="autodiff")
    s_min_var, s_max_var = estimate_sigma_bounds(**kwargs, jacobian_method="variational")

    assert abs(s_max_ad - s_max_var) < 0.05, (
        f"autodiff σ*_max={s_max_ad:.4f} vs variational {s_max_var:.4f}"
    )
    assert abs(s_min_ad - s_min_var) < 0.05, (
        f"autodiff σ*_min={s_min_ad:.4f} vs variational {s_min_var:.4f}"
    )


def test_sigma_bounds_identity_map():
    """For T(x) = x, all eigenvalues should be ≈ 1."""
    torch.manual_seed(7)
    A = torch.tensor([1.0, 1.0])
    v_fn = _linear_v_fn(A)

    # velocity is 0/0 at A=I; define it explicitly as zero
    def zero_v(x, t):
        return torch.zeros_like(x)

    sigma_min_star, sigma_max_star = estimate_sigma_bounds(
        v_fn=zero_v,
        dim=2,
        n_samples=16,
        n_ode_steps=50,
        q_low=0.05,
        q_high=0.95,
        jacobian_method="autodiff",
    )
    # T = identity → J = I → eigenvalues = 1
    assert abs(sigma_max_star - 1.0) < 0.05
    assert abs(sigma_min_star - 1.0) < 0.05
