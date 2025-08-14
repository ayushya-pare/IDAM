# File: compare_idam_vs_adam_rosenbrock.py
# Purpose: Compare IDAM vs custom_Adam (and torch Adam) on Rosenbrock,
#          using the structure of mildlyoverfitted's tutorial.

import math
import numpy as np
import torch
from torch.optim import Adam as TorchAdam
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# Rosenbrock function (same pattern as tutorial)
# -----------------------------
def rosenbrock(xy):
    x, y = xy
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

# -----------------------------
# Your optimizers
# -----------------------------
def adaptive_lr_update(eta_prev: float, disp_norm: float, eps: float,
                       lr_min: float = 1e-3, lr_max: float = 1.0) -> float:
    eta = eta_prev / (math.sqrt(eps + disp_norm))
    return max(min(eta, lr_max), lr_min)

class IDAM(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.001, weight_decay: float = 0.0, update_interval: int = 5):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.update_interval = update_interval
        self.step_counter = 0
        for group in self.param_groups:
            group['eta_adaptive'] = group['lr']
            group['prev_updates'] = [torch.zeros_like(p.data) for p in group['params']]

    @torch.no_grad()
    def step(self):
        self.step_counter += 1
        for group in self.param_groups:
            params, grads = [], []
            for p in group['params']:
                if p.grad is None:
                    continue
                params.append(p)
                grads.append(p.grad)

            if self.step_counter % self.update_interval == 0 and grads:
                flat = [u.view(-1) for u in group['prev_updates'][:len(grads)]]
                disp_norm = torch.cat(flat).norm().item() if flat else 0.0
                group['eta_adaptive'] = adaptive_lr_update(
                    group['eta_adaptive'], disp_norm, eps=1e-8, lr_min=1e-2, lr_max=1.0
                )

            eta = group['eta_adaptive']
            if grads:
                updates = torch._foreach_mul(grads, -eta)
                torch._foreach_add_(params, updates)
                group['prev_updates'] = [u.clone() for u in updates]
        return None

class custom_Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    grad = grad.to_dense()
                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt()).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt()).add_(eps)

                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t
                step_size = lr * (bias_c2 ** 0.5) / bias_c1

                p.addcdiv_(exp_avg, denom, value=-step_size)
        return None

# -----------------------------
# Tutorial-style runner
# -----------------------------
def run_optimization(xy_init, optimizer_class, n_iter, **optimizer_kwargs):
    xy_t = torch.tensor(xy_init, dtype=torch.float32, requires_grad=True)
    opt = optimizer_class([xy_t], **optimizer_kwargs)

    path = np.empty((n_iter + 1, 2), dtype=np.float32)
    path[0, :] = xy_init

    for i in tqdm(range(1, n_iter + 1)):
        opt.zero_grad()
        loss = rosenbrock(xy_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(xy_t, 1.0)  # same as the reference flow
        opt.step()
        path[i, :] = xy_t.detach().numpy()

    return path

def create_animation(paths, colors, names, figsize=(12, 7), x_lim=(-0.1, 1.1), y_lim=(-0.1, 1.1), n_seconds=7):
    if not (len(paths) == len(colors) == len(names)):
        raise ValueError("paths/colors/names length mismatch")

    path_length = max(len(path) for path in paths)
    n_points = 300
    x = np.linspace(*x_lim, n_points)
    y = np.linspace(*y_lim, n_points)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])
    minimum = (1.0, 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.contour(X, Y, Z, 90, cmap="jet")
    scatters = [ax.scatter(None, None, label=label, c=c) for c, label in zip(colors, names)]
    ax.legend(prop={"size": 14})
    ax.plot(*minimum, "rD")

    def animate(i):
        for path, scatter in zip(paths, scatters):
            scatter.set_offsets(path[:i, :])
        ax.set_title(f"iter {i}")

    ms_per_frame = 1000 * n_seconds / path_length
    anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)
    return anim

if __name__ == "__main__":
    # Starting point & iterations (same spirit as the tutorial)
    xy_init = (0.3, 0.8)
    n_iter = 1500

    # Baseline Adam (torch), your custom_Adam, and your IDAM
    path_torch_adam = run_optimization(xy_init, TorchAdam,   n_iter, lr=1e-2)   # a slightly higher LR helps Rosenbrock
    path_custom_adam = run_optimization(xy_init, custom_Adam, n_iter, lr=1e-2)
    path_idam       = run_optimization(xy_init, IDAM,         n_iter, lr=1e-2, update_interval=10)

    # Thinning for visibility
    freq = 10
    paths  = [path_torch_adam[::freq], path_custom_adam[::freq], path_idam[::freq]]
    colors = ["green", "blue", "black"]
    names  = ["torch.Adam", "custom_Adam", "IDAM"]

    anim = create_animation(paths, colors, names, figsize=(12, 7),
                            x_lim=(-0.1, 1.1), y_lim=(-0.1, 1.1), n_seconds=7)
    anim.save("results/2D_Loss/rosenbrock_result.gif")
    #print("results/2D_Loss/rosenbrock_result.gif")

    # Quick textual check of last coordinates
    print("Last 10 steps (custom_Adam):")
    print(path_custom_adam[-10:])
    print("Last 10 steps (IDAM):")
    print(path_idam[-10:])
