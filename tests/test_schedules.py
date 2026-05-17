import torch

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ScheduledConditionalFlowMatcher,
)
from torchcfm.schedules import IdentitySchedule, LearnableMonotoneSchedule, LogisticSchedule


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
