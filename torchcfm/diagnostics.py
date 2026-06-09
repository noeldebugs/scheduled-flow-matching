"""Diagnostics for scheduled conditional flow matching."""

import torch


def estimate_velocity_jacobian_frobenius(model, x, t):
    """Estimate ||d v_theta(x,t) / dx||_F exactly using autograd.

    This is intended for low-dimensional toy experiments.
    """
    x = x.detach().requires_grad_(True)
    t = t.detach()

    v = model(x, t) if callable(model) else model.forward(x, t)

    batch_size = x.shape[0]
    v_flat = v.reshape(batch_size, -1)

    norms = []
    for i in range(v_flat.shape[1]):
        grad_i = torch.autograd.grad(
            v_flat[:, i].sum(),
            x,
            retain_graph=True,
            create_graph=False,
        )[0]
        norms.append(grad_i.reshape(batch_size, -1))

    jac = torch.stack(norms, dim=1)
    frob = torch.linalg.norm(jac, dim=(1, 2))
    return frob


def sample_schedule(schedule, num_points=200, device="cpu"):
    t = torch.linspace(0.0, 1.0, num_points, device=device)
    with torch.no_grad():
        tau = schedule.tau(t)
        tau_dot = schedule.tau_dot(t)
    return t.cpu(), tau.cpu(), tau_dot.cpu()
