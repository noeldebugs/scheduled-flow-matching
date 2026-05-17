import math

import pytest
import torch

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ScheduledConditionalFlowMatcher,
)
from torchcfm.schedules import (
    ClosedFormTauInftySchedule,
    IdentitySchedule,
    LearnableMonotoneSchedule,
    LogisticSchedule,
    tau_infty,
    tau_infty_dot,
)


def test_identity_schedule_reproduces_conditional_flow_matcher():
    x0 = torch.randn(16, 2)
    x1 = torch.randn(16, 2)
    t = torch.rand(16)

    base = ConditionalFlowMatcher(sigma=0.0)
    sched = ScheduledConditionalFlowMatcher(sigma=0.0, schedule=IdentitySchedule())

    mu_base = base.compute_mu_t(x0, x1, t)
    mu_sched = sched.compute_mu_t(x0, x1, t)

    ut_base = base.compute_conditional_flow(x0, x1, t, mu_base)
    ut_sched = sched.compute_conditional_flow(x0, x1, t, mu_sched)

    assert torch.allclose(mu_base, mu_sched, atol=1e-6)
    assert torch.allclose(ut_base, ut_sched, atol=1e-6)


def test_logistic_schedule_satisfies_endpoints():
    schedule = LogisticSchedule(k=8.0, c=0.5)
    t = torch.tensor([0.0, 1.0])
    tau = schedule.tau(t)

    assert torch.allclose(tau[0], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(tau[1], torch.tensor(1.0), atol=1e-6)


def test_logistic_schedule_derivative_is_positive():
    schedule = LogisticSchedule(k=8.0, c=0.5)
    t = torch.linspace(0.0, 1.0, 100)
    tau_dot = schedule.tau_dot(t)

    assert torch.all(tau_dot > 0)


def test_scheduled_matcher_uses_logistic_path_and_velocity():
    x0 = torch.randn(16, 2)
    x1 = torch.randn(16, 2)
    t = torch.rand(16)
    schedule = LogisticSchedule(k=8.0, c=0.5)
    matcher = ScheduledConditionalFlowMatcher(sigma=0.0, schedule=schedule)

    sampled_t, xt, ut = matcher.sample_location_and_conditional_flow(x0, x1, t=t)

    tau_t = schedule.tau(t).reshape(-1, 1)
    tau_dot_t = schedule.tau_dot(t).reshape(-1, 1)
    expected_xt = tau_t * x1 + (1.0 - tau_t) * x0
    expected_ut = tau_dot_t * (x1 - x0)

    assert torch.allclose(sampled_t, t)
    assert torch.allclose(xt, expected_xt, atol=1e-6)
    assert torch.allclose(ut, expected_ut, atol=1e-6)


def test_learnable_schedule_is_monotone():
    schedule = LearnableMonotoneSchedule()
    t = torch.linspace(0.0, 1.0, 100)
    tau = schedule.tau(t)

    assert torch.all(tau[1:] >= tau[:-1] - 1e-6)
    assert torch.allclose(tau[0], torch.tensor(0.0), atol=1e-3)
    assert torch.allclose(tau[-1], torch.tensor(1.0), atol=1e-3)


# ---------------------------------------------------------------------------
# Tests for closed-form τ_∞ (Theorem 6)
# ---------------------------------------------------------------------------

# Representative (f*, g*) pairs covering all three cases
_CASE_A_PARAMS = (1.0, -0.5)     # f*>0>g*, t0 ≈ 0.50
_CASE_B_PARAMS = (2.0, 0.5)      # both pos, f* >= -g*  → Case B
_CASE_C_PARAMS = (0.3, -0.9)     # f* < -g* (-g*=0.9) → Case C
_CASE_ALL = [_CASE_A_PARAMS, _CASE_B_PARAMS, _CASE_C_PARAMS]


@pytest.mark.parametrize("f_star,g_star", _CASE_ALL)
def test_tau_infty_boundary_conditions(f_star, g_star):
    t = torch.tensor([0.0, 1.0])
    tau = tau_infty(t, f_star, g_star)
    assert torch.allclose(tau[0], torch.tensor(0.0), atol=1e-6), (
        f"τ(0) = {tau[0].item():.8f} ≠ 0 for f*={f_star}, g*={g_star}"
    )
    assert torch.allclose(tau[1], torch.tensor(1.0), atol=1e-6), (
        f"τ(1) = {tau[1].item():.8f} ≠ 1 for f*={f_star}, g*={g_star}"
    )


@pytest.mark.parametrize("f_star,g_star", _CASE_ALL)
def test_tau_infty_dot_boundary_conditions(f_star, g_star):
    t = torch.tensor([0.0, 1.0])
    tau_dot = tau_infty_dot(t, f_star, g_star)
    assert torch.all(tau_dot >= 0), (
        f"τ̇ negative at endpoints for f*={f_star}, g*={g_star}"
    )


@pytest.mark.parametrize("f_star,g_star", _CASE_ALL)
def test_tau_infty_monotone(f_star, g_star):
    t = torch.linspace(0.0, 1.0, 500)
    tau = tau_infty(t, f_star, g_star)
    assert torch.all(tau[1:] >= tau[:-1] - 1e-6), (
        f"τ_∞ not monotone for f*={f_star}, g*={g_star}"
    )


def test_tau_infty_case_a_continuity_at_t0():
    """At the transition t0 both pieces should agree to within 1e-6."""
    f_star, g_star = _CASE_A_PARAMS
    from torchcfm.schedules import _tau_infty_params
    case, A, B, C, t0, ln_A, ln_BC = _tau_infty_params(f_star, g_star)
    assert case == "A"
    eps = 1e-5
    t_left = torch.tensor([t0 - eps])
    t_right = torch.tensor([t0 + eps])
    tau_left = tau_infty(t_left, f_star, g_star)
    tau_right = tau_infty(t_right, f_star, g_star)
    assert abs(tau_left.item() - tau_right.item()) < 1e-4, (
        f"Discontinuity at t0={t0:.4f}: τ(t0-)={tau_left.item():.6f}, τ(t0+)={tau_right.item():.6f}"
    )


def test_tau_infty_dot_case_a_continuity_at_t0():
    """τ̇ should be continuous at t0 in Case A."""
    f_star, g_star = _CASE_A_PARAMS
    from torchcfm.schedules import _tau_infty_params
    _, _, _, _, t0, _, _ = _tau_infty_params(f_star, g_star)
    eps = 1e-5
    t_left = torch.tensor([t0 - eps])
    t_right = torch.tensor([t0 + eps])
    dot_left = tau_infty_dot(t_left, f_star, g_star)
    dot_right = tau_infty_dot(t_right, f_star, g_star)
    assert abs(dot_left.item() - dot_right.item()) < 1e-3, (
        f"τ̇ discontinuous at t0={t0:.4f}: {dot_left.item():.6f} vs {dot_right.item():.6f}"
    )


@pytest.mark.parametrize("f_star,g_star", _CASE_ALL)
def test_tau_infty_dot_matches_finite_difference(f_star, g_star):
    """Analytic τ̇ should agree with central FD to within 1e-4.

    Uses float64 to avoid float32 cancellation error at small h.
    """
    t = torch.linspace(0.05, 0.95, 50, dtype=torch.float64)
    h = 1e-5
    tau_plus = tau_infty(t + h, f_star, g_star)
    tau_minus = tau_infty(t - h, f_star, g_star)
    fd = (tau_plus - tau_minus) / (2 * h)
    analytic = tau_infty_dot(t, f_star, g_star)
    max_err = (fd - analytic).abs().max().item()
    assert max_err < 1e-4, (
        f"FD vs analytic τ̇ mismatch: max error={max_err:.2e} for f*={f_star}, g*={g_star}"
    )


def test_closed_form_schedule_no_gradient():
    """ClosedFormTauInftySchedule must have zero trainable parameters."""
    sched = ClosedFormTauInftySchedule(1.0, -0.5)
    assert sum(p.numel() for p in sched.parameters()) == 0


def test_closed_form_schedule_matches_functions():
    """ClosedFormTauInftySchedule.tau / tau_dot delegate correctly."""
    f_star, g_star = 1.0, -0.5
    sched = ClosedFormTauInftySchedule(f_star, g_star)
    t = torch.linspace(0.0, 1.0, 100)
    assert torch.allclose(sched.tau(t), tau_infty(t, f_star, g_star))
    assert torch.allclose(sched.tau_dot(t), tau_infty_dot(t, f_star, g_star))


def test_tau_infty_identity_fallback_zero_f():
    """Edge case f*=0 should return identity."""
    t = torch.linspace(0.0, 1.0, 20)
    tau = tau_infty(t, 0.0, -0.5)
    assert torch.allclose(tau, t, atol=1e-6)


def test_tau_infty_identity_fallback_zero_g():
    """Edge case g*=0 should return identity."""
    t = torch.linspace(0.0, 1.0, 20)
    tau = tau_infty(t, 1.0, 0.0)
    assert torch.allclose(tau, t, atol=1e-6)


def test_tau_infty_figure2_qualitative(tmp_path):
    """Reproduce the qualitative shape of Figure 2 (right) and save a plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    params = [
        (1.0, -0.5, "Case A: f*=1, g*=-0.5"),
        (2.0, 0.5, "Case B: f*=2, g*=0.5"),
        (0.3, -0.9, "Case C: f*=0.3, g*=-0.9"),
    ]
    t = torch.linspace(0.0, 1.0, 200)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for f_star, g_star, label in params:
        tau = tau_infty(t, f_star, g_star).numpy()
        dot = tau_infty_dot(t, f_star, g_star).numpy()
        axes[0].plot(t.numpy(), tau, label=label)
        axes[1].plot(t.numpy(), dot, label=label)
    axes[0].set_title("τ_∞(t)")
    axes[0].legend(fontsize=8)
    axes[1].set_title("τ̇_∞(t)")
    axes[1].legend(fontsize=8)
    out = tmp_path / "tau_infty_fig2.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    assert out.exists()
