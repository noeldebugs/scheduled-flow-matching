"""Estimate σ*_min, σ*_max from a learned flow map.

These bounds are the inputs to the closed-form τ_∞ schedule (Theorem 6).

Public API
----------
estimate_sigma_bounds(v_fn, dim, n_samples, ...) -> (sigma_min_star, sigma_max_star)

The function is standalone — no dependency on any trainer class — so it can
be imported and unit-tested in isolation.
"""

import logging
from typing import Callable, Tuple

import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ODE integration helpers
# ---------------------------------------------------------------------------


def _rk4_integrate(
    v_fn: Callable,
    x0: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    """RK4 integration of dX/dt = v_fn(X, t), preserving the autograd graph.

    Parameters
    ----------
    v_fn   : callable(x: Tensor[bs, d], t: Tensor[bs]) -> Tensor[bs, d]
    x0     : Tensor[bs, d], initial condition (may require_grad)
    n_steps: number of RK4 steps

    Returns
    -------
    X(1) : Tensor[bs, d]
    """
    x = x0
    dt = x0.new_tensor(1.0 / n_steps)
    for step in range(n_steps):
        t_val = step / n_steps
        t = x0.new_tensor(t_val)
        t_b = t.unsqueeze(0).expand(x.shape[0])

        k1 = v_fn(x, t_b)

        t_mid = x0.new_tensor(t_val + 0.5 / n_steps)
        t_mid_b = t_mid.unsqueeze(0).expand(x.shape[0])
        k2 = v_fn(x + (dt * 0.5) * k1, t_mid_b)
        k3 = v_fn(x + (dt * 0.5) * k2, t_mid_b)

        t_end = x0.new_tensor((step + 1) / n_steps)
        t_end_b = t_end.unsqueeze(0).expand(x.shape[0])
        k4 = v_fn(x + dt * k3, t_end_b)

        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


# ---------------------------------------------------------------------------
# Jacobian helpers
# ---------------------------------------------------------------------------


def _jacobian_autodiff(
    v_fn: Callable,
    x0_single: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    """Compute ∂T/∂x0 for one sample via autodiff through the RK4 graph.

    Parameters
    ----------
    x0_single : Tensor[d]

    Returns
    -------
    J : Tensor[d, d]
    """

    def _flow(x0_1d: torch.Tensor) -> torch.Tensor:
        return _rk4_integrate(v_fn, x0_1d.unsqueeze(0), n_steps).squeeze(0)

    return torch.autograd.functional.jacobian(
        _flow, x0_single, create_graph=False, strict=False
    )


def _jacobian_variational(
    v_fn: Callable,
    x0_single: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    """Compute ∂T/∂x0 via variational ODE (Euler).

    Integrates dX/dt = v(X,t) and dJ/dt = (∂v/∂x)(X,t) J simultaneously.
    Uses one full Jacobian call of v per step — O(d²) per step, O(1) graph.

    Parameters
    ----------
    x0_single : Tensor[d]

    Returns
    -------
    J : Tensor[d, d]
    """
    d = x0_single.shape[0]
    device = x0_single.device
    dtype = x0_single.dtype
    dt = 1.0 / n_steps

    x = x0_single.detach().unsqueeze(0)  # (1, d) — no grad needed
    J = torch.eye(d, device=device, dtype=dtype)

    for step in range(n_steps):
        t_val = step * dt
        t_b = x0_single.new_tensor([t_val])

        # Compute ∂v/∂x at current x via a small autograd Jacobian call
        x_for_jac = x.squeeze(0).detach().requires_grad_(True)

        def _v_single(x_: torch.Tensor) -> torch.Tensor:
            return v_fn(x_.unsqueeze(0), t_b).squeeze(0)

        with torch.enable_grad():
            J_v = torch.autograd.functional.jacobian(
                _v_single, x_for_jac, create_graph=False, strict=False
            )  # (d, d)

        # Euler step: J ← J + dt * J_v @ J
        J = J + dt * (J_v.detach() @ J)

        # Euler step: x ← x + dt * v(x, t)
        with torch.no_grad():
            v_x = v_fn(x, t_b)
            x = x + dt * v_x

    return J


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def estimate_sigma_bounds(
    v_fn: Callable,
    dim: int,
    n_samples: int = 512,
    device: torch.device = torch.device("cpu"),
    n_ode_steps: int = 100,
    q_low: float = 0.01,
    q_high: float = 0.99,
    jacobian_method: str = "auto",
) -> Tuple[float, float]:
    """Estimate σ*_min and σ*_max from a learned velocity field.

    Integrates the ODE dX/dt = v_fn(X, t) for n_samples source samples,
    computes the Jacobian ∂X(1)/∂x0 per sample, collects eigenvalues, and
    returns robust quantile estimates of the min/max.

    The velocity field v_fn should be the *trained* model whose training
    target was the conditional flow velocity (with any schedule already
    embedded in v — no external τ̇ factor is applied here).

    Parameters
    ----------
    v_fn           : callable(x: Tensor[bs, d], t: Tensor[bs]) -> Tensor[bs, d]
    dim            : spatial dimension d
    n_samples      : number of source samples to integrate
    device         : torch device
    n_ode_steps    : number of RK4 (or Euler for variational) steps
    q_low, q_high  : quantile levels for σ*_min / σ*_max (default 0.01 / 0.99)
    jacobian_method: "auto" (autodiff for d≤10, variational for d>10),
                     "autodiff", or "variational"

    Returns
    -------
    (sigma_min_star, sigma_max_star) : (float, float)
    """
    if jacobian_method == "auto":
        method = "autodiff" if dim <= 10 else "variational"
    else:
        method = jacobian_method

    log.info(
        "estimate_sigma_bounds: n_samples=%d dim=%d method=%s n_steps=%d",
        n_samples,
        dim,
        method,
        n_ode_steps,
    )

    x0_all = torch.randn(n_samples, dim, device=device)

    sigma_max_list: list = []
    sigma_min_list: list = []

    for i in range(n_samples):
        x0_i = x0_all[i]

        if method == "autodiff":
            J = _jacobian_autodiff(v_fn, x0_i, n_ode_steps)
        else:
            J = _jacobian_variational(v_fn, x0_i, n_ode_steps)

        # Eigenvalue computation with fallback to singular values
        # The paper assumes ∇T is symmetric (OT map = gradient of convex fn).
        # In practice: symmetrize and take real eigenvalues; if the imaginary
        # parts are large (genuinely non-symmetric), use singular values.
        J_sym = 0.5 * (J + J.T)
        eig_complex = torch.linalg.eigvals(J_sym)
        max_imag = eig_complex.imag.abs().max().item()

        if max_imag > 1e-3 * eig_complex.real.abs().max().item():
            # Non-symmetric: fall back to singular values
            sv = torch.linalg.svdvals(J)
            sigma_max_i = sv[0].item()
            sigma_min_i = sv[-1].item()
        else:
            evals = eig_complex.real
            sigma_max_i = evals.max().item()
            sigma_min_i = evals.min().item()

        sigma_max_list.append(sigma_max_i)
        sigma_min_list.append(sigma_min_i)

    sigma_max_t = torch.tensor(sigma_max_list, dtype=torch.float64)
    sigma_min_t = torch.tensor(sigma_min_list, dtype=torch.float64)

    log.info(
        "σ_max per-sample: min=%.4f  mean=%.4f  max=%.4f",
        sigma_max_t.min().item(),
        sigma_max_t.mean().item(),
        sigma_max_t.max().item(),
    )
    log.info(
        "σ_min per-sample: min=%.4f  mean=%.4f  max=%.4f",
        sigma_min_t.min().item(),
        sigma_min_t.mean().item(),
        sigma_min_t.max().item(),
    )

    sigma_max_star = float(torch.quantile(sigma_max_t, q_high))
    sigma_min_star = float(torch.quantile(sigma_min_t, q_low))

    log.info("σ*_max (q=%.2f) = %.4f", q_high, sigma_max_star)
    log.info("σ*_min (q=%.2f) = %.4f", q_low, sigma_min_star)

    return sigma_min_star, sigma_max_star
