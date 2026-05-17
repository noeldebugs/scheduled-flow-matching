"""Tests for time schedules and scheduled Conditional Flow Matching.

Covers:
1. Equivalence: IdentitySchedule reproduces vanilla CFM exactly.
2. Schedule properties: boundary conditions, monotonicity, tau_dot correctness.
3. Guard tests: incompatible classes raise NotImplementedError.
4. Smoke training: stable loss for IdentitySchedule and SigmoidSchedule.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.schedules import IdentitySchedule, Schedule, SigmoidSchedule

SEED = 42
BS = 64


# ---------------------------------------------------------------------------
# 1. Equivalence: IdentitySchedule must match vanilla CFM exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "FM_cls,kwargs",
    [
        (ConditionalFlowMatcher, {"sigma": 0.0}),
        (ConditionalFlowMatcher, {"sigma": 0.1}),
        (ExactOptimalTransportConditionalFlowMatcher, {"sigma": 0.0}),
    ],
)
def test_identity_schedule_matches_vanilla(FM_cls, kwargs):
    """IdentitySchedule must reproduce vanilla CFM output to float32 precision."""
    torch.manual_seed(SEED)
    x0 = torch.randn(BS, 2)
    x1 = torch.randn(BS, 2)

    fm_vanilla = FM_cls(**kwargs)
    fm_identity = FM_cls(**kwargs, schedule=IdentitySchedule())

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_v, xt_v, ut_v, eps_v = fm_vanilla.sample_location_and_conditional_flow(
        x0, x1, return_noise=True
    )
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_i, xt_i, ut_i, eps_i = fm_identity.sample_location_and_conditional_flow(
        x0, x1, return_noise=True
    )

    assert torch.allclose(t_v, t_i), "t mismatch with IdentitySchedule"
    assert torch.allclose(xt_v, xt_i, atol=1e-6), "xt mismatch with IdentitySchedule"
    assert torch.allclose(ut_v, ut_i, atol=1e-6), "ut mismatch with IdentitySchedule"
    assert torch.allclose(eps_v, eps_i), "eps mismatch with IdentitySchedule"


def test_identity_schedule_matches_vanilla_sb():
    """IdentitySchedule on SB-CFM must match unscheduled SB-CFM."""
    torch.manual_seed(SEED)
    import numpy as np

    np.random.seed(SEED)
    x0 = torch.randn(BS, 2)
    x1 = torch.randn(BS, 2)
    sigma = 0.5

    fm_vanilla = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, ot_method="sinkhorn")
    fm_identity = SchrodingerBridgeConditionalFlowMatcher(
        sigma=sigma, ot_method="sinkhorn", schedule=IdentitySchedule()
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_v, xt_v, ut_v, eps_v = fm_vanilla.sample_location_and_conditional_flow(
        x0, x1, return_noise=True
    )
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_i, xt_i, ut_i, eps_i = fm_identity.sample_location_and_conditional_flow(
        x0, x1, return_noise=True
    )

    assert torch.allclose(t_v, t_i)
    assert torch.allclose(xt_v, xt_i, atol=1e-6)
    assert torch.allclose(ut_v, ut_i, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Schedule property tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schedule",
    [
        IdentitySchedule(),
        SigmoidSchedule(k=1.0),
        SigmoidSchedule(k=5.0),
        SigmoidSchedule(k=10.0),
    ],
)
def test_schedule_boundary_conditions(schedule):
    """tau(0) == 0 and tau(1) == 1 exactly."""
    t0 = torch.tensor([0.0])
    t1 = torch.tensor([1.0])
    assert torch.allclose(schedule.tau(t0), torch.zeros(1), atol=1e-6), "tau(0) != 0"
    assert torch.allclose(schedule.tau(t1), torch.ones(1), atol=1e-6), "tau(1) != 1"


@pytest.mark.parametrize(
    "schedule",
    [
        IdentitySchedule(),
        SigmoidSchedule(k=1.0),
        SigmoidSchedule(k=5.0),
        SigmoidSchedule(k=10.0),
    ],
)
def test_schedule_monotonicity(schedule):
    """tau is non-decreasing and tau_dot is non-negative."""
    t = torch.linspace(0.0, 1.0, 1001)
    tau = schedule.tau(t)
    tau_dot = schedule.tau_dot(t)

    assert (torch.diff(tau) >= -1e-7).all(), "tau is not monotone"
    assert (tau_dot >= -1e-7).all(), "tau_dot is negative"


@pytest.mark.parametrize(
    "schedule",
    [
        IdentitySchedule(),
        SigmoidSchedule(k=1.0),
        SigmoidSchedule(k=5.0),
        SigmoidSchedule(k=10.0),
    ],
)
def test_schedule_tau_dot_matches_finite_difference(schedule):
    """tau_dot must match the centered finite difference of tau to 1e-4.

    Uses float64 to avoid float32 cancellation errors in the subtraction,
    which otherwise dominate at ~1e-3 and make the check unreliable.
    """
    t = torch.linspace(0.01, 0.99, 999, dtype=torch.float64)
    dt = 1e-6
    fd = (schedule.tau(t + dt) - schedule.tau(t - dt)) / (2 * dt)
    tau_dot = schedule.tau_dot(t)
    assert torch.allclose(fd, tau_dot, atol=1e-4), "tau_dot deviates from finite differences"


def test_sigmoid_schedule_k_validation():
    """SigmoidSchedule must reject k <= 0."""
    with pytest.raises(ValueError, match="k must be strictly positive"):
        SigmoidSchedule(k=0.0)
    with pytest.raises(ValueError, match="k must be strictly positive"):
        SigmoidSchedule(k=-1.0)


def test_schedule_is_nn_module():
    """Schedule subclasses must be nn.Module instances."""
    assert isinstance(IdentitySchedule(), nn.Module)
    assert isinstance(SigmoidSchedule(), nn.Module)


def test_schedule_vectorized_over_batch():
    """Schedules must handle arbitrary leading batch dimensions."""
    t = torch.rand(32, 1, 1)
    for schedule in [IdentitySchedule(), SigmoidSchedule(k=5.0)]:
        assert schedule.tau(t).shape == t.shape
        assert schedule.tau_dot(t).shape == t.shape


# ---------------------------------------------------------------------------
# 3. Guard tests for incompatible flow matchers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "FM_cls",
    [TargetConditionalFlowMatcher, VariancePreservingConditionalFlowMatcher],
)
def test_incompatible_classes_raise(FM_cls):
    """VP-CFM and Target-CFM must raise NotImplementedError with a non-identity schedule."""
    with pytest.raises(NotImplementedError):
        FM_cls(sigma=0.0, schedule=SigmoidSchedule(k=5.0))


@pytest.mark.parametrize(
    "FM_cls",
    [TargetConditionalFlowMatcher, VariancePreservingConditionalFlowMatcher],
)
def test_incompatible_classes_allow_identity(FM_cls):
    """VP-CFM and Target-CFM must accept IdentitySchedule without raising."""
    FM_cls(sigma=0.0, schedule=IdentitySchedule())
    FM_cls(sigma=0.0, schedule=None)


# ---------------------------------------------------------------------------
# 4. Smoke training run
# ---------------------------------------------------------------------------


class _TinyMLP(nn.Module):
    """Minimal MLP: input (x, t) → velocity, used only in smoke tests."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32), nn.SELU(), nn.Linear(32, 32), nn.SELU(), nn.Linear(32, 2)
        )

    def forward(self, xt, t):
        inp = torch.cat([xt, t.unsqueeze(-1)], dim=-1)
        return self.net(inp)


def _run_smoke_training(schedule, n_steps=50, batch_size=128, seed=SEED):
    """Train a tiny MLP for n_steps on 2D Gaussians; return final loss."""
    torch.manual_seed(seed)
    fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0, schedule=schedule)
    model = _TinyMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(n_steps):
        x0 = torch.randn(batch_size, 2)
        x1 = torch.randn(batch_size, 2) + 4.0
        t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
        vt = model(xt, t)
        loss = torch.mean((vt - ut) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item()


@pytest.mark.slow
def test_smoke_training_identity():
    """IdentitySchedule: training must complete with finite loss."""
    loss = _run_smoke_training(IdentitySchedule())
    assert math.isfinite(loss), f"Loss is not finite: {loss}"


@pytest.mark.slow
def test_smoke_training_sigmoid():
    """SigmoidSchedule: training must complete with finite loss."""
    loss = _run_smoke_training(SigmoidSchedule(k=5.0))
    assert math.isfinite(loss), f"Loss is not finite: {loss}"


def test_smoke_training_fast(monkeypatch):
    """Fast (non-marked) smoke check: 10 steps, finite loss for both schedules."""
    for schedule in [IdentitySchedule(), SigmoidSchedule(k=5.0)]:
        loss = _run_smoke_training(schedule, n_steps=10)
        assert math.isfinite(loss), f"Non-finite loss with {type(schedule).__name__}: {loss}"
