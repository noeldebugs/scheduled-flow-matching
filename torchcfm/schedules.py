"""Monotone schedules for conditional flow matching paths."""

import logging
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class BaseSchedule(nn.Module):
    """Base class for monotone schedules tau: [0, 1] -> [0, 1]."""

    def tau(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def tau_dot(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.tau(t)


class IdentitySchedule(BaseSchedule):
    """tau(t) = t."""

    def tau(self, t):
        return t

    def tau_dot(self, t):
        return torch.ones_like(t)


class PowerSchedule(BaseSchedule):
    """tau(t) = t ** alpha."""

    def __init__(self, alpha: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = eps

    def tau(self, t):
        t_safe = torch.clamp(t, self.eps, 1.0)
        return t_safe.pow(self.alpha)

    def tau_dot(self, t):
        t_safe = torch.clamp(t, self.eps, 1.0)
        return self.alpha * t_safe.pow(self.alpha - 1.0)


class LogisticSchedule(BaseSchedule):
    """Normalized sigmoid schedule.

    tau(t) = [sigmoid(k(t-c)) - sigmoid(-kc)]
             / [sigmoid(k(1-c)) - sigmoid(-kc)].
    """

    def __init__(self, k: float = 8.0, c: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.k = float(k)
        self.c = float(c)
        self.eps = eps

    def _constants(self, t):
        k = torch.as_tensor(self.k, dtype=t.dtype, device=t.device)
        c = torch.as_tensor(self.c, dtype=t.dtype, device=t.device)
        left = torch.sigmoid(-k * c)
        right = torch.sigmoid(k * (1.0 - c))
        denom = torch.clamp(right - left, min=self.eps)
        return k, c, left, denom

    def tau(self, t):
        k, c, left, denom = self._constants(t)
        val = torch.sigmoid(k * (t - c))
        return (val - left) / denom

    def tau_dot(self, t):
        k, c, _, denom = self._constants(t)
        val = torch.sigmoid(k * (t - c))
        return k * val * (1.0 - val) / denom


class LearnableMonotoneSchedule(BaseSchedule):
    """Learnable monotone schedule using normalized positive speed."""

    def __init__(
        self,
        hidden_dim: int = 32,
        num_grid: int = 256,
        eps: float = 1e-4,
        identity_init: bool = True,
    ):
        super().__init__()
        self.num_grid = num_grid
        self.eps = eps

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        if identity_init:
            for module in self.net.modules():
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)

    def raw_speed(self, t):
        return F.softplus(self.net(t.reshape(-1, 1))).reshape_as(t) + self.eps

    def _grid(self, device, dtype):
        return torch.linspace(0.0, 1.0, self.num_grid, device=device, dtype=dtype)

    def _cdf_grid(self, device, dtype):
        grid = self._grid(device, dtype)
        speed = self.raw_speed(grid)

        dt = grid[1] - grid[0]
        increments = 0.5 * (speed[:-1] + speed[1:]) * dt
        cdf = torch.cat(
            [torch.zeros(1, device=device, dtype=dtype), torch.cumsum(increments, dim=0)]
        )
        cdf = cdf / torch.clamp(cdf[-1], min=self.eps)
        return grid, cdf, speed, cdf[-1]

    def tau(self, t):
        t_flat = t.reshape(-1)
        grid, cdf, _, _ = self._cdf_grid(t.device, t.dtype)

        idx = torch.searchsorted(grid, t_flat.clamp(0.0, 1.0), right=True) - 1
        idx = idx.clamp(0, self.num_grid - 2)

        t0 = grid[idx]
        t1 = grid[idx + 1]
        y0 = cdf[idx]
        y1 = cdf[idx + 1]

        w = (t_flat - t0) / torch.clamp(t1 - t0, min=self.eps)
        out = y0 + w * (y1 - y0)
        return out.reshape_as(t)

    def tau_dot(self, t):
        grid = self._grid(t.device, t.dtype)
        speed_grid = self.raw_speed(grid)
        dt = grid[1] - grid[0]
        integral = torch.sum(0.5 * (speed_grid[:-1] + speed_grid[1:]) * dt)
        return self.raw_speed(t) / torch.clamp(integral, min=self.eps)

    def smoothness_regularizer(self):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        grid = self._grid(device, dtype)
        speed = self.raw_speed(grid)
        return torch.mean((speed[1:] - speed[:-1]) ** 2)

    def speed_l2_regularizer(self):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        grid = self._grid(device, dtype)
        speed = self.raw_speed(grid)
        return torch.mean(speed**2)


# ---------------------------------------------------------------------------
# Closed-form τ_∞ schedule  (Theorem 6, Tsimpos, Ren, Zech & Marzouk 2024)
# ---------------------------------------------------------------------------

_EPS = 1e-8


def _tau_infty_params(f_star: float, g_star: float):
    """Pre-compute the scalar constants for tau_infty / tau_infty_dot.

    Returns (case, A, B, C, t0, lnA, lnBC) where case ∈ {'A','B','C','Id'}.
    Only the fields relevant to the returned case are meaningful; the rest are 0.

    Inputs
    ------
    f_star : σ*_max − 1  (sup of max-eigenvalue of ∇T, minus 1)
    g_star : σ*_min − 1  (inf of min-eigenvalue of ∇T, minus 1)

    Assumption: σ*_min > 0, i.e. g_star > −1.
    """
    if g_star <= -1.0 + _EPS:
        warnings.warn(
            f"g_star={g_star:.4f} ≤ −1 (σ*_min ≤ 0); falling back to identity schedule.",
            stacklevel=3,
        )
        return ("Id", 0.0, 0.0, 0.0, 0.5, 0.0, 0.0)

    if abs(f_star) < _EPS or abs(g_star) < _EPS:
        log.debug("f_star or g_star ≈ 0; using identity schedule.")
        return ("Id", 0.0, 0.0, 0.0, 0.5, 0.0, 0.0)

    # A = (1/(4(g*+1))) * (2 - f*/g* - g*/f*)   [same A used in t0 and piece-1]
    inner = 2.0 - f_star / g_star - g_star / f_star
    A = inner / (4.0 * (g_star + 1.0))

    # Attempt to compute t0 = ln[(1/2)(1 - f*/g*)] / ln(A)
    val_for_ln = 0.5 * (1.0 - f_star / g_star)
    t0 = float("nan")
    if A > _EPS and val_for_ln > _EPS:
        ln_A = math.log(A)
        if abs(ln_A) > _EPS:
            t0 = math.log(val_for_ln) / ln_A

    if not math.isnan(t0) and 0.0 <= t0 <= 1.0:
        # Case A: piecewise exponential (eq. 19 + appendix eq. 108)
        # Piece-1: (1/f*) A^t − 1/f*
        # Piece-2: (1/2)(1/g* − 1/f*) B^t C^(1−t) − 1/g*
        #   B = 2(g*+1)/(1 − g*/f*)          [paper eq. 19 and appendix]
        #   C = (1/2)(1 − f*/g*)              [appendix eq. 108 — NOT (1−g*/f*)/2]
        B = 2.0 * (g_star + 1.0) / (1.0 - g_star / f_star)
        C = 0.5 * (1.0 - f_star / g_star)    # appendix formula; C = val_for_ln
        ln_A = math.log(A)
        # ln(B/C) used in derivative of piece-2
        ln_BC = math.log(B / C) if B > _EPS and C > _EPS else 0.0
        return ("A", A, B, C, t0, ln_A, ln_BC)

    # No transition in [0,1]: choose Case B or C
    if f_star >= -g_star:
        # Case B: τ(t) = ((f*+1)^t − 1) / f*
        return ("B", 0.0, 0.0, 0.0, float("nan"), 0.0, 0.0)
    else:
        # Case C: τ(t) = ((g*+1)^t − 1) / g*
        return ("C", 0.0, 0.0, 0.0, float("nan"), 0.0, 0.0)


def tau_infty(t: torch.Tensor, f_star: float, g_star: float) -> torch.Tensor:
    """Closed-form optimal schedule τ_∞(t) from Theorem 6.

    Parameters
    ----------
    t       : Tensor  — time values in [0, 1]
    f_star  : float   — σ*_max − 1  (sup of Jacobian max-eigenvalue minus 1)
    g_star  : float   — σ*_min − 1  (inf of Jacobian min-eigenvalue minus 1)

    Returns
    -------
    tau : Tensor, same shape as t, values in [0, 1]
    """
    case, A, B, C, t0, ln_A, ln_BC = _tau_infty_params(f_star, g_star)

    if case == "Id":
        return t.clone()

    if case == "A":
        # Piece 1 for t ≤ t0: (1/f*) A^t − 1/f*
        piece1 = (A**t) / f_star - 1.0 / f_star
        # Piece 2 for t ≥ t0: (1/2)(1/g* − 1/f*) B^t C^(1−t) − 1/g*
        piece2 = 0.5 * (1.0 / g_star - 1.0 / f_star) * (B**t) * (C ** (1.0 - t)) - 1.0 / g_star
        return torch.where(t <= t0, piece1, piece2)

    if case == "B":
        # ((f*+1)^t − 1) / f*
        return ((f_star + 1.0) ** t - 1.0) / f_star

    # case == "C"
    # ((g*+1)^t − 1) / g*
    return ((g_star + 1.0) ** t - 1.0) / g_star


def tau_infty_dot(t: torch.Tensor, f_star: float, g_star: float) -> torch.Tensor:
    """Analytic derivative τ̇_∞(t) of the closed-form optimal schedule.

    Parameters
    ----------
    t       : Tensor  — time values in [0, 1]
    f_star  : float   — σ*_max − 1
    g_star  : float   — σ*_min − 1

    Returns
    -------
    tau_dot : Tensor, same shape as t, values ≥ 0
    """
    case, A, B, C, t0, ln_A, ln_BC = _tau_infty_params(f_star, g_star)

    if case == "Id":
        return torch.ones_like(t)

    if case == "A":
        # d/dt [(1/f*) A^t − 1/f*] = (ln A / f*) A^t
        dot1 = (ln_A / f_star) * (A**t)
        # d/dt [(1/2)(1/g*−1/f*) B^t C^(1−t) − 1/g*]
        #   = (1/2)(1/g*−1/f*) B^t C^(1−t) ln(B/C)
        dot2 = 0.5 * (1.0 / g_star - 1.0 / f_star) * (B**t) * (C ** (1.0 - t)) * ln_BC
        return torch.where(t <= t0, dot1, dot2)

    if case == "B":
        # d/dt [((f*+1)^t − 1) / f*] = ln(f*+1) (f*+1)^t / f*
        ln_fp1 = math.log(f_star + 1.0)
        return ln_fp1 * (f_star + 1.0) ** t / f_star

    # case == "C"
    # d/dt [((g*+1)^t − 1) / g*] = ln(g*+1) (g*+1)^t / g*
    ln_gp1 = math.log(g_star + 1.0)
    return ln_gp1 * (g_star + 1.0) ** t / g_star


class ClosedFormTauInftySchedule(BaseSchedule):
    """Frozen closed-form optimal schedule τ_∞ (Theorem 6).

    Stores (f_star, g_star) as plain floats — no nn.Parameters, so zero
    gradients flow into the schedule.  Drop-in replacement for any BaseSchedule.

    Parameters
    ----------
    f_star : float — σ*_max − 1
    g_star : float — σ*_min − 1
    """

    def __init__(self, f_star: float, g_star: float) -> None:
        super().__init__()
        self.f_star = float(f_star)
        self.g_star = float(g_star)
        case, _, _, _, t0, _, _ = _tau_infty_params(f_star, g_star)
        self._case = case
        self._t0 = t0  # transition time (nan for cases B/C)
        log.info(
            "ClosedFormTauInftySchedule: f*=%.4f g*=%.4f case=%s t0=%s",
            f_star,
            g_star,
            case,
            f"{t0:.4f}" if not math.isnan(t0) else "n/a",
        )

    def tau(self, t: torch.Tensor) -> torch.Tensor:
        return tau_infty(t, self.f_star, self.g_star)

    def tau_dot(self, t: torch.Tensor) -> torch.Tensor:
        return tau_infty_dot(t, self.f_star, self.g_star)

    def extra_repr(self) -> str:
        t0_str = f"{self._t0:.4f}" if not math.isnan(self._t0) else "n/a"
        return f"f_star={self.f_star:.4f}, g_star={self.g_star:.4f}, case={self._case}, t0={t0_str}"
