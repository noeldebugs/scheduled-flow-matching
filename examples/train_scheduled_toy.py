"""Train scheduled conditional flow matching on a 2D toy dataset.

Example
-------
# Fixed or learnable schedules (approach a):
python examples/train_scheduled_toy.py --steps 5000 --schedule logistic --matcher ot

# Iterative closed-form τ_∞ schedule (approach b):
python examples/train_scheduled_toy.py --schedule closed_form_iter --n-outer 3
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torchcfm.conditional_flow_matching import (
    ScheduledConditionalFlowMatcher,
    ScheduledExactOptimalTransportConditionalFlowMatcher,
)
from torchcfm.diagnostics import sample_schedule
from torchcfm.schedules import (
    ClosedFormTauInftySchedule,
    IdentitySchedule,
    LearnableMonotoneSchedule,
    LogisticSchedule,
    PowerSchedule,
)
from torchcfm.sigma_estimation import estimate_sigma_bounds
from torchcfm.utils import sample_1d_crossing, sample_1d_gmm, sample_8gaussians, sample_moons

log = logging.getLogger(__name__)


class TimeConditionedMLP(nn.Module):
    """Small velocity model v_theta(x, t) for 2D toy data."""

    def __init__(self, dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x, t):
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        if t.dim() == 1:
            t = t[:, None]
        t = t.to(device=x.device, dtype=x.dtype)
        return self.net(torch.cat([x, t], dim=-1))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument(
        "--target",
        choices=["8gaussians", "moons", "1d_crossing", "1d_gmm"],
        default="8gaussians",
    )
    parser.add_argument("--matcher", choices=["ot", "independent"], default="ot")
    parser.add_argument(
        "--ot-method",
        choices=["exact", "sinkhorn", "unbalanced", "partial"],
        default="exact",
    )
    parser.add_argument("--sigma", type=float, default=0.0)

    parser.add_argument(
        "--schedule",
        choices=["identity", "power", "logistic", "learnable", "closed_form_iter"],
        default="logistic",
    )
    parser.add_argument("--power-alpha", type=float, default=1.0)
    parser.add_argument("--logistic-k", type=float, default=8.0)
    parser.add_argument("--logistic-c", type=float, default=0.5)
    parser.add_argument("--schedule-hidden-dim", type=int, default=32)
    parser.add_argument("--schedule-grid", type=int, default=256)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--speed-l2-weight", type=float, default=0.0)

    # Closed-form iterative schedule (approach b) options
    parser.add_argument("--n-outer", type=int, default=3, help="outer iterations for closed_form_iter")
    parser.add_argument("--steps-per-iter", type=int, default=None, help="steps per outer iter (default: same as --steps)")
    parser.add_argument("--sigma-n-samples", type=int, default=256, help="samples for sigma bound estimation")
    parser.add_argument("--sigma-n-ode-steps", type=int, default=100, help="ODE steps for sigma estimation")
    parser.add_argument("--sigma-q-low", type=float, default=0.01)
    parser.add_argument("--sigma-q-high", type=float, default=0.99)

    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--integration-steps", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("results/scheduled_toy"))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    return parser.parse_args()


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_schedule(args):
    if args.schedule == "identity":
        return IdentitySchedule()
    if args.schedule == "power":
        return PowerSchedule(alpha=args.power_alpha)
    if args.schedule == "logistic":
        return LogisticSchedule(k=args.logistic_k, c=args.logistic_c)
    if args.schedule == "learnable":
        return LearnableMonotoneSchedule(
            hidden_dim=args.schedule_hidden_dim,
            num_grid=args.schedule_grid,
        )
    if args.schedule == "closed_form_iter":
        return IdentitySchedule()  # starting point; replaced by outer loop
    raise ValueError(f"Unknown schedule: {args.schedule}")


def build_matcher(args, schedule):
    if args.matcher == "independent":
        return ScheduledConditionalFlowMatcher(sigma=args.sigma, schedule=schedule)
    if args.matcher == "ot":
        return ScheduledExactOptimalTransportConditionalFlowMatcher(
            sigma=args.sigma,
            schedule=schedule,
            ot_method=args.ot_method,
        )
    raise ValueError(f"Unknown matcher: {args.matcher}")


_TARGET_DIM = {
    "8gaussians": 2,
    "moons": 2,
    "1d_crossing": 1,
    "1d_gmm": 1,
}


def sample_target(name, batch_size, device):
    if name == "8gaussians":
        return sample_8gaussians(batch_size).to(device)
    if name == "moons":
        return sample_moons(batch_size).float().to(device)
    if name == "1d_crossing":
        return sample_1d_crossing(batch_size).to(device)
    if name == "1d_gmm":
        return sample_1d_gmm(batch_size).to(device)
    raise ValueError(f"Unknown target dataset: {name}")


def schedule_regularization(schedule, smoothness_weight, speed_l2_weight):
    reg = 0.0
    if smoothness_weight > 0 and hasattr(schedule, "smoothness_regularizer"):
        reg = reg + smoothness_weight * schedule.smoothness_regularizer()
    if speed_l2_weight > 0 and hasattr(schedule, "speed_l2_regularizer"):
        reg = reg + speed_l2_weight * schedule.speed_l2_regularizer()
    return reg


@torch.no_grad()
def euler_integrate(model, x0, num_steps):
    x = x0.clone()
    t_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=x.device, dtype=x.dtype)
    for i in range(num_steps):
        t = t_grid[i].expand(x.shape[0])
        dt = t_grid[i + 1] - t_grid[i]
        x = x + dt * model(x, t)
    return x


def _train_inner(args, model, schedule, n_steps, device, prefix="", dim=2):
    """Train model with a fixed (possibly frozen) schedule for n_steps.

    Optimises only model.parameters(); schedule parameters (if any) are also
    included so that learnable schedules still work when called from train().
    """
    matcher = build_matcher(args, schedule)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(schedule.parameters()),
        lr=args.lr,
    )

    model.train()
    for step in range(1, n_steps + 1):
        x0 = torch.randn(args.batch_size, dim, device=device)
        x1 = sample_target(args.target, args.batch_size, device)

        t, xt, ut = matcher.sample_location_and_conditional_flow(x0, x1)
        pred = model(xt, t)
        flow_loss = torch.mean((pred - ut) ** 2)
        reg_loss = schedule_regularization(
            schedule,
            smoothness_weight=args.smoothness_weight,
            speed_l2_weight=args.speed_l2_weight,
        )
        loss = flow_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            tag = f"[{prefix}] " if prefix else ""
            print(
                f"{tag}step {step:05d} | loss {loss.item():.6f} | "
                f"flow {flow_loss.item():.6f}"
            )

    return model, schedule


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    dim = _TARGET_DIM[args.target]

    schedule = build_schedule(args).to(device)
    model = TimeConditionedMLP(dim=dim, hidden_dim=args.hidden_dim).to(device)
    return _train_inner(args, model, schedule, args.steps, device, dim=dim)


def train_closed_form_iter(args):
    """Outer loop for approach (b): iterative closed-form τ_∞ scheduling."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    dim = _TARGET_DIM[args.target]
    steps_per_iter = args.steps_per_iter if args.steps_per_iter is not None else args.steps

    model = TimeConditionedMLP(dim=dim, hidden_dim=args.hidden_dim).to(device)
    schedule = IdentitySchedule().to(device)
    all_schedules = [schedule]

    # ── Iteration 0: train with τ = Identity ──────────────────────────────
    print(f"\n{'='*60}")
    print("Outer iteration 0 — τ = Identity")
    print(f"{'='*60}")
    model, schedule = _train_inner(args, model, schedule, args.steps, device, prefix="iter-0", dim=dim)

    for outer_iter in range(1, args.n_outer + 1):
        # ── Estimate σ bounds from current flow ───────────────────────────
        print(f"\nEstimating σ bounds (outer iter {outer_iter})…")
        model.eval()

        def v_fn(x, t):
            return model(x, t)

        sigma_min_star, sigma_max_star = estimate_sigma_bounds(
            v_fn=v_fn,
            dim=dim,
            n_samples=args.sigma_n_samples,
            device=device,
            n_ode_steps=args.sigma_n_ode_steps,
            q_low=args.sigma_q_low,
            q_high=args.sigma_q_high,
        )

        f_star = sigma_max_star - 1.0
        g_star = sigma_min_star - 1.0
        print(
            f"  σ*_max={sigma_max_star:.4f}  σ*_min={sigma_min_star:.4f}  "
            f"f*={f_star:.4f}  g*={g_star:.4f}"
        )

        # ── Build new closed-form schedule ────────────────────────────────
        schedule = ClosedFormTauInftySchedule(f_star, g_star).to(device)
        all_schedules.append(schedule)
        t0_str = f"{schedule._t0:.4f}" if not (schedule._t0 != schedule._t0) else "n/a"
        print(f"  case={schedule._case}  t0={t0_str}")

        # ── Retrain v with frozen τ_∞ ─────────────────────────────────────
        n_steps = steps_per_iter if outer_iter < args.n_outer else args.steps
        print(f"\n{'='*60}")
        print(f"Outer iteration {outer_iter} — τ = ClosedForm (case {schedule._case})")
        print(f"{'='*60}")
        model.train()
        model, schedule = _train_inner(
            args, model, schedule, n_steps, device, prefix=f"iter-{outer_iter}", dim=dim
        )

    return model, schedule, all_schedules


def _plot_schedule_overlay(schedules, output_path):
    """Save τ(t) and τ̇(t) overlaid for all schedules across iterations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_np = torch.linspace(0.0, 1.0, 200)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    cmap = plt.cm.viridis
    for i, sched in enumerate(schedules):
        label = f"iter {i}" + (" (Id)" if i == 0 else "")
        color = cmap(i / max(len(schedules) - 1, 1))
        with torch.no_grad():
            tau = sched.tau(t_np).cpu().numpy()
            tau_dot = sched.tau_dot(t_np).cpu().numpy()
        axes[0].plot(t_np.numpy(), tau, label=label, color=color)
        axes[1].plot(t_np.numpy(), tau_dot, label=label, color=color)
    axes[0].set_title("τ(t)")
    axes[0].legend(fontsize=8)
    axes[1].set_title("τ̇(t)")
    axes[1].legend(fontsize=8)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_lipschitz_proxy(model, device, output_path, dim=2, n_eval=512, n_t=50):
    """Save t ↦ mean Frobenius norm of ∂v/∂x estimated on random samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchcfm.diagnostics import estimate_velocity_jacobian_frobenius

    t_vals = torch.linspace(0.0, 1.0, n_t, device=device)
    x = torch.randn(n_eval, dim, device=device)
    frob_means = []
    model.eval()
    for t_val in t_vals:
        t_b = t_val.expand(n_eval)
        frob = estimate_velocity_jacobian_frobenius(model, x, t_b)
        frob_means.append(frob.mean().item())
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(t_vals.cpu().numpy(), frob_means)
    ax.set_xlabel("t")
    ax.set_ylabel("mean ‖∂v/∂x‖_F")
    ax.set_title("Spatial Lipschitz proxy")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_outputs(args, model, schedule, device, all_schedules=None):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    dim = _TARGET_DIM[args.target]
    x0 = torch.randn(args.eval_samples, dim, device=device)
    x1 = sample_target(args.target, args.eval_samples, device)
    generated = euler_integrate(model, x0, args.integration_steps)

    if args.checkpoint_path is not None:
        args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model": model.state_dict(),
            "args": vars(args),
        }
        if hasattr(schedule, "state_dict"):
            ckpt["schedule"] = schedule.state_dict()
        if isinstance(schedule, ClosedFormTauInftySchedule):
            ckpt["f_star"] = schedule.f_star
            ckpt["g_star"] = schedule.g_star
        torch.save(ckpt, args.checkpoint_path)

    if args.no_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0 = x0.cpu()
    x1 = x1.cpu()
    generated = generated.cpu()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    if dim == 1:
        bins = 80
        rng = (min(x0[:, 0].min(), x1[:, 0].min(), generated[:, 0].min()).item() - 0.5,
               max(x0[:, 0].max(), x1[:, 0].max(), generated[:, 0].max()).item() + 0.5)
        for ax, data, title in [
            (axes[0], x0, "source"),
            (axes[1], generated, "generated"),
            (axes[2], x1, "target"),
        ]:
            ax.hist(data[:, 0].numpy(), bins=bins, range=rng, density=True, alpha=0.7)
            ax.set_title(title)
            ax.set_yticks([])
    else:
        for ax, data, title in [
            (axes[0], x0, "source"),
            (axes[1], generated, "generated"),
            (axes[2], x1, "target"),
        ]:
            ax.scatter(data[:, 0], data[:, 1], s=3, alpha=0.5)
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(args.output_dir / "samples.png", dpi=200)
    plt.close(fig)

    t, tau, tau_dot = sample_schedule(schedule, device=device)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    axes[0].plot(t, tau)
    axes[0].set_title("tau(t)")
    axes[1].plot(t, tau_dot)
    axes[1].set_title("tau_dot(t)")
    fig.savefig(args.output_dir / "schedule.png", dpi=200)
    plt.close(fig)

    # Closed-form iter diagnostics: overlay of all iteration schedules
    if all_schedules is not None and len(all_schedules) > 1:
        _plot_schedule_overlay(all_schedules, args.output_dir / "schedule_overlay.png")
        _plot_lipschitz_proxy(model, device, args.output_dir / "lipschitz_proxy.png", dim=dim)


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    device = resolve_device(args.device)

    if args.schedule == "closed_form_iter":
        model, schedule, all_schedules = train_closed_form_iter(args)
        save_outputs(args, model, schedule, device, all_schedules=all_schedules)
    else:
        model, schedule = train(args)
        save_outputs(args, model, schedule, device)

    print(f"saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
