# Time Scheduling for Conditional Flow Matching

Based on Tsimpos, Ren, Zech & Marzouk, *"Optimal Scheduling for Conditional Flow Matching"* (2024).

## What scheduling is

Standard CFM samples `t ~ U[0,1]` and trains on the interpolant

```
X_t = t·X_1 + (1-t)·X_0,    target u_t = X_1 - X_0
```

With a **time schedule** `τ: [0,1] → [0,1]` (monotone, `τ(0)=0`, `τ(1)=1`) the interpolant becomes

```
X_t^τ = τ(t)·X_1 + (1-τ(t))·X_0,    target u_τ = τ̇(t)·(X_1 - X_0)
```

`t` is still drawn uniformly; only the speed of traversal changes. The model `v_θ(x, t)` absorbs `τ̇`, so the ODE integrator is unchanged.

**Why bother?** The optimal `τ` is sigmoid-like — slow near the endpoints, fast in the middle. This makes the spatial Lipschitz constant of the velocity field roughly uniform in time, which yields exponentially better worst-case approximation error for the same model capacity.

The optimality guarantees apply to **OT-CFM**. For SB-CFM the same reparameterization is a reasonable heuristic.

## When to use it

- **Use a schedule** when you care about approximation quality per parameter, or when training with OT-CFM and want the theoretical benefits of the paper.
- **Leave it at the default** (`IdentitySchedule`) when reproducing prior results or using `TargetConditionalFlowMatcher` / `VariancePreservingConditionalFlowMatcher` (incompatible interpolants).

## API

```python
from torchcfm.schedules import IdentitySchedule, SigmoidSchedule
```

### `IdentitySchedule`

`τ(t) = t`, `τ̇(t) = 1`. Default everywhere. Reproduces vanilla CFM exactly.

### `SigmoidSchedule(k=5.0)`

```
τ(t)  = [σ(k·(t − ½)) − σ(−k/2)] / [σ(k/2) − σ(−k/2)]
τ̇(t) = k·σ(k·(t−½))·(1−σ(k·(t−½))) / [σ(k/2) − σ(−k/2)]
```

`σ(x) = 1/(1+e^{-x})`. Endpoints are exact. The denominator `tanh(k/4)` is never zero.

| `k` | character |
|-----|-----------|
| 1   | mild warp |
| 5   | moderate (paper's recommended range) |
| 10  | strong, nearly step-like near endpoints |

Validate `k > 0` is enforced at construction time.

## Minimal example

```python
import torch
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.schedules import SigmoidSchedule

FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0, schedule=SigmoidSchedule(k=5.0))

x0 = torch.randn(256, 2)
x1 = torch.randn(256, 2) + 4.0

t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
# ut is now τ̇(t)·(x1 − x0) instead of (x1 − x0)
# The call site and loss are otherwise unchanged:
# loss = F.mse_loss(v_theta(xt, t), ut)
```

## Compatibility

| Class | Schedule support |
|---|---|
| `ConditionalFlowMatcher` | ✓ all schedules |
| `ExactOptimalTransportConditionalFlowMatcher` | ✓ all schedules (paper's primary target) |
| `SchrodingerBridgeConditionalFlowMatcher` | ✓ heuristic extension (see note below) |
| `TargetConditionalFlowMatcher` | ✗ raises `NotImplementedError` |
| `VariancePreservingConditionalFlowMatcher` | ✗ raises `NotImplementedError` |

**SB-CFM note**: the bridge width is reparameterized as `σ·√(τ(t)·(1−τ(t)))` to match the scheduled interpolant. This is geometrically consistent but the paper's optimality guarantees do not apply.

## Writing a custom schedule

Subclass `Schedule` (an `nn.Module`) and implement `tau` and `tau_dot`:

```python
from torchcfm.schedules import Schedule
import torch
from torch import Tensor

class PowerSchedule(Schedule):
    def __init__(self, p: float = 2.0):
        super().__init__()
        self.p = p

    def tau(self, t: Tensor) -> Tensor:
        return t ** self.p

    def tau_dot(self, t: Tensor) -> Tensor:
        return self.p * t ** (self.p - 1)
```

Requirements: `τ(0) = 0`, `τ(1) = 1`, `τ̇(t) ≥ 0`, differentiable in `t`.

## Iterative refinement (paper §3.3)

The paper describes a closed-form optimal `τ_∞` derived from Lipschitz bounds estimated on a pilot run with `IdentitySchedule`. This is not implemented. To approximate it: train with `IdentitySchedule`, estimate the time-varying Lipschitz constant of `v_θ`, then derive `τ` from those bounds and retrain.
