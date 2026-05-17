"""Time schedules for Conditional Flow Matching.

Implements the Schedule abstraction from:
    Tsimpos, Ren, Zech & Marzouk, "Optimal Scheduling for Conditional Flow Matching" (2024).

A schedule τ: [0,1] → [0,1] reparameterizes the CFM interpolant:

    X_t^τ  = τ(t)·X₁ + (1 - τ(t))·X₀
    u_τ(X_t^τ | X₀, X₁) = τ̇(t)·(X₁ - X₀)

Sampling t ~ U[0,1] and using (X_t^τ, τ̇·(X₁-X₀)) is equivalent to
sampling t' ~ τ_#U[0,1] and using the standard linear interpolant.
The optimal τ is sigmoid-like, which makes the spatial Lipschitz constant
of the velocity field roughly uniform in time.
"""

import math
from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class Schedule(nn.Module):
    """Abstract base class for time schedules τ: [0,1] → [0,1].

    A schedule is a monotone map with τ(0) = 0 and τ(1) = 1. Subclass
    this and implement ``tau`` and ``tau_dot``. Inheriting from nn.Module
    keeps parameters accessible for future learnable schedules.

    Parameters satisfy:
        - τ(0) = 0, τ(1) = 1 (boundary conditions)
        - τ̇(t) ≥ 0  (monotonicity)
        - τ is differentiable in t (required for training)
    """

    @abstractmethod
    def tau(self, t: Tensor) -> Tensor:
        """Forward map τ(t).

        Parameters
        ----------
        t : Tensor, shape (...)
            Time values, typically in [0, 1].

        Returns
        -------
        Tensor, same shape as t
        """
        raise NotImplementedError

    @abstractmethod
    def tau_dot(self, t: Tensor) -> Tensor:
        """Derivative τ̇(t) = dτ/dt.

        Parameters
        ----------
        t : Tensor, shape (...)
            Time values, typically in [0, 1].

        Returns
        -------
        Tensor, same shape as t. Non-negative everywhere.
        """
        raise NotImplementedError

    def forward(self, t: Tensor) -> Tensor:
        """Alias for ``tau(t)`` for nn.Module compatibility."""
        return self.tau(t)


class IdentitySchedule(Schedule):
    """Identity schedule: τ(t) = t, τ̇(t) = 1.

    Reproduces vanilla CFM exactly and is the default everywhere a schedule
    is accepted. Use this as the backward-compatibility anchor — any result
    computed with IdentitySchedule must match the unscheduled implementation
    to numerical precision.
    """

    def tau(self, t: Tensor) -> Tensor:
        return t

    def tau_dot(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)


class SigmoidSchedule(Schedule):
    """Normalized logit-sigmoid schedule with slow-at-endpoints traversal.

    Implements the time-warping family from Tsimpos et al. (2024) §3. The
    schedule traverses slowly near t = 0 and t = 1 and rapidly through the
    middle, which makes the spatial Lipschitz constant of the optimal
    velocity field roughly uniform in time and yields exponentially better
    worst-case approximation for OT-CFM. For SB-CFM the same reparameterization
    is a reasonable heuristic extension; the paper's optimality guarantees
    apply to OT-CFM only.

    Definition::

        τ(t)   = [σ(k·(t − ½)) − σ(−k/2)] / [σ(k/2) − σ(−k/2)]
        τ̇(t)  = k·σ(k·(t − ½))·[1 − σ(k·(t − ½))] / [σ(k/2) − σ(−k/2)]

    where σ(x) = 1/(1 + exp(−x)). The denominator equals tanh(k/4), so
    it is never zero for finite k. The boundary conditions τ(0) = 0,
    τ(1) = 1 are satisfied exactly.

    Hook for iterative refinement (paper §3.3, option b): the closed-form
    optimal τ_∞ can be derived from Lipschitz bounds estimated on a pilot
    run with IdentitySchedule. That procedure is not implemented here; use
    this class's k parameter to set sharpness manually.

    Parameters
    ----------
    k : float
        Sharpness of the schedule. k = 5–10 is the regime studied in the
        paper. Larger k → faster middle / slower endpoints. Must be > 0.
    """

    def __init__(self, k: float = 5.0):
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be strictly positive, got {k}.")
        self.k = float(k)
        # Precomputed as Python floats: device-agnostic scalar arithmetic.
        # _offset = σ(−k/2),  _normalizer = σ(k/2) − σ(−k/2) = tanh(k/4).
        self._offset: float = 1.0 / (1.0 + math.exp(self.k / 2))
        self._normalizer: float = math.tanh(self.k / 4)

    def tau(self, t: Tensor) -> Tensor:
        return (torch.sigmoid(self.k * (t - 0.5)) - self._offset) / self._normalizer

    def tau_dot(self, t: Tensor) -> Tensor:
        s = torch.sigmoid(self.k * (t - 0.5))
        return (self.k * s * (1.0 - s)) / self._normalizer
