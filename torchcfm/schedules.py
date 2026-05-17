"""Monotone schedules for conditional flow matching paths."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse softplus."""
    return x + torch.log(-torch.expm1(-x))


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerically stable inverse sigmoid / logit."""
    x = torch.clamp(x, eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


class PowerSchedule(BaseSchedule):
    """Learnable power schedule.

    tau(t) = t ** alpha

    alpha is learned but constrained to be positive.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        eps: float = 1e-5,
        min_alpha: float = 1e-4,
        learnable: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.min_alpha = min_alpha

        alpha_init = torch.tensor(float(alpha) - min_alpha)
        alpha_init = torch.clamp(alpha_init, min=eps)

        raw_alpha = _inverse_softplus(alpha_init)

        if learnable:
            self.raw_alpha = nn.Parameter(raw_alpha)
        else:
            self.register_buffer("raw_alpha", raw_alpha)

    @property
    def alpha_value(self):
        return F.softplus(self.raw_alpha) + self.min_alpha

    def tau(self, t):
        t_safe = torch.clamp(t, self.eps, 1.0)
        alpha = self.alpha_value.to(dtype=t.dtype, device=t.device)
        return t_safe.pow(alpha)

    def tau_dot(self, t):
        t_safe = torch.clamp(t, self.eps, 1.0)
        alpha = self.alpha_value.to(dtype=t.dtype, device=t.device)
        return alpha * t_safe.pow(alpha - 1.0)


class LogisticSchedule(BaseSchedule):
    """Learnable normalized sigmoid schedule.

    tau(t) = [sigmoid(k(t-c)) - sigmoid(-kc)]
             / [sigmoid(k(1-c)) - sigmoid(-kc)]

    k is learned but constrained to be positive.
    c is learned but constrained to lie in (0, 1).
    """

    def __init__(
        self,
        k: float = 8.0,
        c: float = 0.5,
        eps: float = 1e-8,
        min_k: float = 1e-4,
        learnable: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.min_k = min_k

        k_init = torch.tensor(float(k) - min_k)
        k_init = torch.clamp(k_init, min=eps)
        raw_k = _inverse_softplus(k_init)

        c_init = torch.tensor(float(c))
        raw_c = _inverse_sigmoid(c_init)

        if learnable:
            self.raw_k = nn.Parameter(raw_k)
            self.raw_c = nn.Parameter(raw_c)
        else:
            self.register_buffer("raw_k", raw_k)
            self.register_buffer("raw_c", raw_c)

    @property
    def k_value(self):
        return F.softplus(self.raw_k) + self.min_k

    @property
    def c_value(self):
        return torch.sigmoid(self.raw_c)

    def _constants(self, t):
        k = self.k_value.to(dtype=t.dtype, device=t.device)
        c = self.c_value.to(dtype=t.dtype, device=t.device)

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

        normalizer = torch.clamp(cdf[-1], min=self.eps)
        cdf = cdf / normalizer

        return grid, cdf, speed, normalizer

    def tau(self, t):
        t_flat = t.reshape(-1)
        grid, cdf, _, _ = self._cdf_grid(t.device, t.dtype)

        t_clamped = t_flat.clamp(0.0, 1.0)

        idx = torch.searchsorted(grid, t_clamped, right=True) - 1
        idx = idx.clamp(0, self.num_grid - 2)

        t0 = grid[idx]
        t1 = grid[idx + 1]
        y0 = cdf[idx]
        y1 = cdf[idx + 1]

        w = (t_clamped - t0) / torch.clamp(t1 - t0, min=self.eps)
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
