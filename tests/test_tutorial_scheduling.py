"""Scheduling feature on the tutorial distributions.

Tests the Schedule abstraction against the setups from the bundled tutorial notebooks:

- ``examples/2D_tutorials/1D_Gaussian_mixture_FM.ipynb``
  Source p0: N(0, 1)
  Target p1: ½N(−2, 0.01) + ½N(+2, 0.01)

- ``examples/2D_tutorials/1D_Crossing_Flows_ICFM_vs_OTCFM.ipynb``
  Source p0: ½N(−3, 0.04) + ½N(+3, 0.04)
  Target p1: ½N(−1, 0.04) + ½N(+1, 0.04)

Test groups
-----------
1. Convergence  — all schedules produce finite loss on both setups.
2. Equivalence  — IdentitySchedule matches vanilla CFM to float32 precision.
3. Velocity scaling — SigmoidSchedule multiplies targets by τ̇(t) correctly.
4. OT coupling  — coupling benefit holds with and without scheduling (slow).
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchcfm.schedules import IdentitySchedule, SigmoidSchedule

SEED = 42


# ---------------------------------------------------------------------------
# Tutorial distributions
# ---------------------------------------------------------------------------


def _sample_bimodal(n: int, centers: list[float], std: float) -> torch.Tensor:
    component = torch.randint(0, 2, (n,))
    means = torch.where(component == 0, torch.tensor(centers[0]), torch.tensor(centers[1]))
    return (means + std * torch.randn(n)).unsqueeze(1)


# 1D_Gaussian_mixture_FM  (notebook: sigma=0.01)
def _gm_source(n: int) -> torch.Tensor:
    return torch.randn(n, 1)


def _gm_target(n: int) -> torch.Tensor:
    return _sample_bimodal(n, [-2.0, 2.0], std=0.1)


# 1D_Crossing_Flows_ICFM_vs_OTCFM  (notebook: sigma=0.01)
def _cf_source(n: int) -> torch.Tensor:
    return _sample_bimodal(n, [-3.0, 3.0], std=0.2)


def _cf_target(n: int) -> torch.Tensor:
    return _sample_bimodal(n, [-1.0, 1.0], std=0.2)


# ---------------------------------------------------------------------------
# Minimal 1-D MLP (input: [x, t] → velocity)
# ---------------------------------------------------------------------------


class _MLP1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.SELU(),
            nn.Linear(64, 64), nn.SELU(),
            nn.Linear(64, 1),
        )

    def forward(self, xt, t):
        return self.net(torch.cat([xt, t.unsqueeze(-1)], dim=-1))


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------


def _run_training(fm, source_fn, target_fn, n_steps=200, batch_size=256, seed=SEED):
    """Train _MLP1D for n_steps and return the final loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _MLP1D()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = None
    for _ in range(n_steps):
        x0 = source_fn(batch_size)
        x1 = target_fn(batch_size)
        t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
        vt = model(xt, t)
        loss = torch.mean((vt - ut) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


# ---------------------------------------------------------------------------
# 1. Convergence: all schedules must produce finite loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schedule",
    [
        IdentitySchedule(),
        SigmoidSchedule(k=1.0),
        SigmoidSchedule(k=5.0),
        SigmoidSchedule(k=10.0),
    ],
    ids=["identity", "sigmoid_k1", "sigmoid_k5", "sigmoid_k10"],
)
def test_gm_icfm_finite_loss(schedule):
    """All schedules converge (finite loss) on the Gaussian-mixture tutorial."""
    fm = ConditionalFlowMatcher(sigma=0.01, schedule=schedule)
    loss = _run_training(fm, _gm_source, _gm_target)
    assert math.isfinite(loss), f"Non-finite loss ({type(schedule).__name__}): {loss}"


@pytest.mark.parametrize(
    "schedule",
    [
        IdentitySchedule(),
        SigmoidSchedule(k=5.0),
    ],
    ids=["identity", "sigmoid_k5"],
)
def test_gm_otcfm_finite_loss(schedule):
    """OT-CFM + schedule converges on the Gaussian-mixture tutorial."""
    fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.01, schedule=schedule)
    loss = _run_training(fm, _gm_source, _gm_target)
    assert math.isfinite(loss), f"Non-finite loss ({type(schedule).__name__}): {loss}"


@pytest.mark.parametrize(
    "schedule",
    [
        IdentitySchedule(),
        SigmoidSchedule(k=5.0),
    ],
    ids=["identity", "sigmoid_k5"],
)
def test_cf_otcfm_finite_loss(schedule):
    """OT-CFM + schedule converges on the crossing-flows tutorial."""
    fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.01, schedule=schedule)
    loss = _run_training(fm, _cf_source, _cf_target)
    assert math.isfinite(loss), f"Non-finite loss ({type(schedule).__name__}): {loss}"


# ---------------------------------------------------------------------------
# 2. Equivalence: IdentitySchedule must match vanilla CFM on tutorial data
# ---------------------------------------------------------------------------


def test_identity_matches_vanilla_gm():
    """IdentitySchedule reproduces vanilla I-CFM on Gaussian-mixture data."""
    torch.manual_seed(SEED)
    x0 = _gm_source(256)
    x1 = _gm_target(256)

    fm_v = ConditionalFlowMatcher(sigma=0.01)
    fm_i = ConditionalFlowMatcher(sigma=0.01, schedule=IdentitySchedule())

    torch.manual_seed(SEED)
    t_v, xt_v, ut_v = fm_v.sample_location_and_conditional_flow(x0, x1)
    torch.manual_seed(SEED)
    t_i, xt_i, ut_i = fm_i.sample_location_and_conditional_flow(x0, x1)

    assert torch.allclose(t_v, t_i), "t mismatch"
    assert torch.allclose(xt_v, xt_i, atol=1e-6), "xt mismatch"
    assert torch.allclose(ut_v, ut_i, atol=1e-6), "ut mismatch"


def test_identity_matches_vanilla_cf():
    """IdentitySchedule reproduces vanilla OT-CFM on crossing-flows data."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    x0 = _cf_source(256)
    x1 = _cf_target(256)

    fm_v = ExactOptimalTransportConditionalFlowMatcher(sigma=0.01)
    fm_i = ExactOptimalTransportConditionalFlowMatcher(sigma=0.01, schedule=IdentitySchedule())

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_v, xt_v, ut_v = fm_v.sample_location_and_conditional_flow(x0, x1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_i, xt_i, ut_i = fm_i.sample_location_and_conditional_flow(x0, x1)

    assert torch.allclose(t_v, t_i), "t mismatch"
    assert torch.allclose(xt_v, xt_i, atol=1e-6), "xt mismatch"
    assert torch.allclose(ut_v, ut_i, atol=1e-6), "ut mismatch"


# ---------------------------------------------------------------------------
# 3. Velocity scaling: SigmoidSchedule applies τ̇(t) to the velocity target
# ---------------------------------------------------------------------------


def test_velocity_target_scaled_by_tau_dot():
    """Velocity target ut = τ̇(t)·(x1−x0); with SigmoidSchedule, ut at t≈0 is smaller."""
    torch.manual_seed(SEED)
    x0 = _gm_source(512)
    x1 = _gm_target(512)

    sigma = 0.0  # no noise so xt is deterministic
    fm_id = ConditionalFlowMatcher(sigma=sigma)
    fm_sig = ConditionalFlowMatcher(sigma=sigma, schedule=SigmoidSchedule(k=5.0))

    # Fix t near 0 where τ̇ < 1 for SigmoidSchedule
    t_near_zero = torch.full((512,), 0.05)

    _, _, ut_id = fm_id.sample_location_and_conditional_flow(x0, x1, t=t_near_zero)
    _, _, ut_sig = fm_sig.sample_location_and_conditional_flow(x0, x1, t=t_near_zero)

    # τ̇(0.05) < 1, so scheduled velocity magnitude should be smaller
    tau_dot = SigmoidSchedule(k=5.0).tau_dot(torch.tensor([0.05])).item()
    assert tau_dot < 1.0, "τ̇ should be < 1 near t=0 for sigmoid schedule"
    assert ut_sig.abs().mean() < ut_id.abs().mean(), (
        "Scheduled velocity should be smaller near t=0 than identity velocity"
    )


def test_velocity_target_scaled_near_midpoint():
    """At t=0.5, SigmoidSchedule's τ̇ > 1, so |ut_sig| > |ut_id|."""
    torch.manual_seed(SEED)
    x0 = _gm_source(512)
    x1 = _gm_target(512)

    sigma = 0.0
    fm_id = ConditionalFlowMatcher(sigma=sigma)
    fm_sig = ConditionalFlowMatcher(sigma=sigma, schedule=SigmoidSchedule(k=5.0))

    t_mid = torch.full((512,), 0.5)

    _, _, ut_id = fm_id.sample_location_and_conditional_flow(x0, x1, t=t_mid)
    _, _, ut_sig = fm_sig.sample_location_and_conditional_flow(x0, x1, t=t_mid)

    tau_dot_mid = SigmoidSchedule(k=5.0).tau_dot(torch.tensor([0.5])).item()
    assert tau_dot_mid > 1.0, "τ̇(0.5) should be > 1 for sigmoid schedule"
    assert ut_sig.abs().mean() > ut_id.abs().mean(), (
        "Scheduled velocity should be larger at t=0.5 than identity velocity"
    )


def test_velocity_target_ratio_matches_tau_dot():
    """ut_sig / ut_id ≈ τ̇(t) at every sample point (no noise path)."""
    torch.manual_seed(SEED)
    x0 = _gm_source(256)
    x1 = _gm_target(256)

    schedule = SigmoidSchedule(k=5.0)
    fm_id = ConditionalFlowMatcher(sigma=0.0)
    fm_sig = ConditionalFlowMatcher(sigma=0.0, schedule=schedule)

    t = torch.rand(256, generator=torch.Generator().manual_seed(SEED))

    _, _, ut_id = fm_id.sample_location_and_conditional_flow(x0, x1, t=t.clone())
    _, _, ut_sig = fm_sig.sample_location_and_conditional_flow(x0, x1, t=t.clone())

    tau_dot = schedule.tau_dot(t).unsqueeze(-1)  # (256, 1)
    expected_sig = tau_dot * ut_id  # τ̇(t) · (x1 − x0)

    assert torch.allclose(ut_sig, expected_sig, atol=1e-5), (
        "Scheduled ut should equal τ̇(t) × unscheduled ut"
    )


# ---------------------------------------------------------------------------
# 4. OT coupling benefit — with and without scheduling (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_otcfm_beats_icfm_crossing_flows():
    """OT coupling achieves lower training loss than I-CFM on crossing flows.

    Reproduces the core finding of the 1D_Crossing_Flows tutorial:
    the monotone OT plan eliminates crossing trajectories and reduces
    training loss compared to independent coupling.
    """
    n_steps, batch_size = 2000, 512
    loss_icfm = _run_training(
        ConditionalFlowMatcher(sigma=0.01),
        _cf_source, _cf_target,
        n_steps=n_steps, batch_size=batch_size,
    )
    loss_otcfm = _run_training(
        ExactOptimalTransportConditionalFlowMatcher(sigma=0.01),
        _cf_source, _cf_target,
        n_steps=n_steps, batch_size=batch_size,
    )
    assert loss_otcfm < loss_icfm, (
        f"OT-CFM ({loss_otcfm:.4f}) should beat I-CFM ({loss_icfm:.4f}) on crossing flows"
    )


@pytest.mark.slow
def test_scheduled_otcfm_comparable_to_unscheduled():
    """OT-CFM + SigmoidSchedule achieves comparable loss to unscheduled OT-CFM.

    The schedule reparameterises time but doesn't change the modelled distribution —
    both should converge to low loss on the Gaussian-mixture tutorial.
    """
    n_steps, batch_size = 2000, 512
    loss_vanilla = _run_training(
        ExactOptimalTransportConditionalFlowMatcher(sigma=0.01),
        _gm_source, _gm_target,
        n_steps=n_steps, batch_size=batch_size,
    )
    loss_scheduled = _run_training(
        ExactOptimalTransportConditionalFlowMatcher(sigma=0.01, schedule=SigmoidSchedule(k=5.0)),
        _gm_source, _gm_target,
        n_steps=n_steps, batch_size=batch_size,
    )
    # Both should be finite and in a comparable range (within 2×)
    assert math.isfinite(loss_scheduled), f"Scheduled loss not finite: {loss_scheduled}"
    assert loss_scheduled < 2 * loss_vanilla, (
        f"Scheduled loss ({loss_scheduled:.4f}) much higher than vanilla ({loss_vanilla:.4f})"
    )
