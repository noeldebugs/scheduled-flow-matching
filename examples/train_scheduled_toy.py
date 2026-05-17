"""Train scheduled conditional flow matching on a 2D toy dataset.

Example
-------
python examples/train_scheduled_toy.py --steps 5000 --schedule logistic --matcher ot
"""

import argparse
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
    IdentitySchedule,
    LearnableMonotoneSchedule,
    LogisticSchedule,
    PowerSchedule,
)
from torchcfm.utils import sample_8gaussians, sample_moons


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

    parser.add_argument("--target", choices=["8gaussians", "moons"], default="8gaussians")
    parser.add_argument("--matcher", choices=["ot", "independent"], default="ot")
    parser.add_argument(
        "--ot-method",
        choices=["exact", "sinkhorn", "unbalanced", "partial"],
        default="exact",
    )
    parser.add_argument("--sigma", type=float, default=0.0)

    parser.add_argument(
        "--schedule",
        choices=["identity", "power", "logistic", "learnable"],
        default="logistic",
    )
    parser.add_argument("--power-alpha", type=float, default=1.0)
    parser.add_argument("--logistic-k", type=float, default=8.0)
    parser.add_argument("--logistic-c", type=float, default=0.5)
    parser.add_argument("--schedule-hidden-dim", type=int, default=32)
    parser.add_argument("--schedule-grid", type=int, default=256)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--speed-l2-weight", type=float, default=0.0)

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


def sample_target(name, batch_size, device):
    if name == "8gaussians":
        return sample_8gaussians(batch_size).to(device)
    if name == "moons":
        return sample_moons(batch_size).float().to(device)
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


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    schedule = build_schedule(args).to(device)
    matcher = build_matcher(args, schedule)
    model = TimeConditionedMLP(hidden_dim=args.hidden_dim).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(schedule.parameters()),
        lr=args.lr,
    )

    model.train()
    for step in range(1, args.steps + 1):
        x0 = torch.randn(args.batch_size, 2, device=device)
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
            print(f"step {step:05d} | loss {loss.item():.6f} | flow {flow_loss.item():.6f}")

    return model, schedule


def save_outputs(args, model, schedule, device):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    x0 = torch.randn(args.eval_samples, 2, device=device)
    x1 = sample_target(args.target, args.eval_samples, device)
    generated = euler_integrate(model, x0, args.integration_steps)

    if args.checkpoint_path is not None:
        args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "schedule": schedule.state_dict(),
                "args": vars(args),
            },
            args.checkpoint_path,
        )

    if args.no_plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0 = x0.cpu()
    x1 = x1.cpu()
    generated = generated.cpu()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
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


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model, schedule = train(args)
    save_outputs(args, model, schedule, device)
    print(f"saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

# Test all schedules on the 8 Gaussians target with OT matching.
# python examples/train_scheduled_toy.py --steps 5000 --schedule identity --matcher ot --target 8gaussians
# python examples/train_scheduled_toy.py --steps 5000 --schedule power --matcher ot --target 8gaussians
# python examples/train_scheduled_toy.py --steps 5000 --schedule logistic --matcher ot --target 8gaussians
# python examples/train_scheduled_toy.py --steps 5000 --schedule learnable --matcher ot --target 8gaussians

# Test all schedules on the two half moons target with OT matching.
# python examples/train_scheduled_toy.py --steps 5000 --schedule identity --matcher ot --target moons
# python examples/train_scheduled_toy.py --steps 5000 --schedule power --matcher ot --target moons
# python examples/train_scheduled_toy.py --steps 5000 --schedule logistic --matcher ot --target moons
# python examples/train_scheduled_toy.py --steps 5000 --schedule learnable --matcher ot --target moons

# Compare OT versus independent matching on 8 Gaussians using the learnable logistic schedule.
# python examples/train_scheduled_toy.py --steps 5000 --schedule logistic --matcher ot --target 8gaussians
# python examples/train_scheduled_toy.py --steps 5000 --schedule logistic --matcher independent --target 8gaussians

# Compare OT versus independent matching on moons using the learnable logistic schedule.
# python examples/train_scheduled_toy.py --steps 5000 --schedule logistic --matcher ot --target moons
# python examples/train_scheduled_toy.py --steps 5000 --schedule logistic --matcher independent --target moons

# Test learnable power schedule with different initial alpha values on moons.
# python examples/train_scheduled_toy.py --steps 5000 --schedule power --power-alpha 1.0 --matcher ot --target moons
# python examples/train_scheduled_toy.py --steps 5000 --schedule power --power-alpha 2.0 --matcher ot --target moons
# python examples/train_scheduled_toy.py --steps 5000 --schedule power --power-alpha 0.5 --matcher ot --target moons

# Test learnable logistic schedule with different initial steepness values on moons.
# python examples/train_scheduled_toy.py --steps 5000 --schedule logistic --logistic-k 2.0 --logistic-c 0.5 --matcher ot --target moons
# python examples/train_scheduled_toy.py --steps 5000 --schedule logistic --logistic-k 8.0 --logistic-c 0.5 --matcher ot --target moons
# python examples/train_scheduled_toy.py --steps 5000 --schedule logistic --logistic-k 16.0 --logistic-c 0.5 --matcher ot --target moons

# Test learnable logistic schedule with different initial center locations on moons.
# python examples/train_scheduled_toy.py --steps 5000 --schedule logistic --logistic-k 8.0 --logistic-c 0.35 --matcher ot --target moons
# python examples/train_scheduled_toy.py --steps 5000 --schedule logistic --logistic-k 8.0 --logistic-c 0.65 --matcher ot --target moons
