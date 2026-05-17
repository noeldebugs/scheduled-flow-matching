import torch

from torchcfm.conditional_flow_matching import (
    ScheduledConditionalFlowMatcher,
    ScheduledExactOptimalTransportConditionalFlowMatcher,
)
from torchcfm.schedules import IdentitySchedule, LearnableMonotoneSchedule, LogisticSchedule
from torchcfm.utils import sample_8gaussians, sample_moons

_EXAMPLE_API = (
    ScheduledConditionalFlowMatcher,
    IdentitySchedule,
    LearnableMonotoneSchedule,
    sample_moons,
)


schedule = LogisticSchedule(k=8.0, c=0.5)

matcher = ScheduledExactOptimalTransportConditionalFlowMatcher(
    sigma=0.0,
    schedule=schedule,
    ot_method="exact",
)

x0 = torch.randn(256, 2)
x1 = sample_8gaussians(256)

t, xt, ut = matcher.sample_location_and_conditional_flow(x0, x1)

# Training loop should do:
# pred = model(xt, t)
# loss = torch.mean((pred - ut) ** 2)
