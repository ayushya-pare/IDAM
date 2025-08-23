# File: compare_idam_vs_custom_adam_toy.py

import math, time, argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# Repro
# -----------------------------
def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# custom_Adam (from scratch)
# -----------------------------
class custom_Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0, amsgrad=False):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            lr, (b1,b2), eps, wd, ams = g['lr'], g['betas'], g['eps'], g['weight_decay'], g['amsgrad']
            for p in g['params']:
                if p.grad is None: 
                    continue
                grad = p.grad.to_dense() if p.grad.is_sparse else p.grad
                if wd: 
                    grad = grad.add(p, alpha=wd)

                s = self.state[p]
                if len(s) == 0:
                    s['step'] = 0
                    s['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    s['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if ams:
                        s['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                s['step'] += 1
                t = s['step']

                s['exp_avg'].mul_(b1).add_(grad, alpha=1-b1)
                s['exp_avg_sq'].mul_(b2).addcmul_(grad, grad, value=1-b2)

                denom_t = s['max_exp_avg_sq'] if ams else s['exp_avg_sq']
                if ams:
                    torch.maximum(denom_t, s['exp_avg_sq'], out=denom_t)

                denom = denom_t.sqrt().add_(eps)
                step_size = lr * ((1 - b2**t)**0.5) / (1 - b1**t)
                p.addcdiv_(s['exp_avg'], denom, value=-step_size)
        return None

# -----------------------------
# Improved IDAM
# -----------------------------
class IDAM(torch.optim.Optimizer):
    """
    - Scalar adaptive LR per param group
    - LR_t = clip( base_lr / sqrt(eps + disp_ema^2), lr_min, lr_max )
    - disp_ema is an EMA of the L2 norm of the previous updates' displacement
    - Optional foreach fast path (math is the same)
    """
    def __init__(
        self,
        params,
        lr: float = 0.1,          # higher base LR works better here than Adam's 1e-3
        weight_decay: float = 0.0, # kept for API parity; not applied
        update_interval: int = 1,  # adapt every step
        lr_min: float = 1e-2,
        lr_max: float = 0.5,
        beta_disp: float = 0.9,   # EMA factor for displacement norm
        foreach: bool = False
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay, update_interval=update_interval,
                        lr_min=lr_min, lr_max=lr_max, beta_disp=beta_disp, foreach=foreach)
        super().__init__(params, defaults)

        self.step_counter = 0
        for group in self.param_groups:
            group['eta_adaptive'] = group['lr']
            group['disp_ema'] = 0.0
            group['prev_updates'] = [torch.zeros_like(p.data) for p in group['params']]

    @torch.no_grad()
    def step(self):
        self.step_counter += 1

        for group in self.param_groups:
            params_with_grad, grads, idxs = [], [], []
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    g = g.to_dense()
                params_with_grad.append(p); grads.append(g); idxs.append(i)

            # Adapt LR every k steps using EMA of displacement norm
            if self.step_counter % group['update_interval'] == 0 and idxs:
                flat_prev = [group['prev_updates'][i].view(-1) for i in idxs]
                disp_norm = torch.cat(flat_prev).norm().item() if flat_prev else 0.0
#                d_ema = group['disp_ema'] = group['beta_disp'] * group['disp_ema'] + (1 - group['beta_disp']) * disp_norm
                eta = group['lr'] / math.sqrt(1e-8 + disp_norm*disp_norm)
                eta = max(min(eta, group['lr_max']), group['lr_min'])
                group['eta_adaptive'] = eta

            eta = group['eta_adaptive']

            if group['foreach'] and grads:
                updates = torch._foreach_mul(grads, -eta)
                torch._foreach_add_(params_with_grad, updates)
                group['prev_updates'] = [u.clone() for u in updates]
            else:
                for p, g, i in zip(params_with_grad, grads, idxs):
                    p.add_(g, alpha=-eta)
                    group['prev_updates'][i] = (-eta * g).clone()
        return None

# -----------------------------
# Simple MLP for the toy task
# -----------------------------
class BigMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1], bias=True))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Synthetic dataset generator (CPU tensors; batches moved to device)
# -----------------------------
def make_synthetic_dataset(n_train=12000, n_val=3000, in_dim=1024, n_classes=10, seed=123):
    g = torch.Generator().manual_seed(seed)
    W_star = torch.randn(in_dim, n_classes, generator=g)   # CPU
    b_star = torch.randn(n_classes, generator=g)

    def gen_split(n):
        X = torch.randn(n, in_dim, generator=g)            # CPU
        logits = X @ W_star + b_star
        y = logits.argmax(dim=1)
        return X, y

    Xtr, ytr = gen_split(n_train)
    Xva, yva = gen_split(n_val)
    return TensorDataset(Xtr, ytr), TensorDataset(Xva, yva)

@torch.no_grad()
def eval_accuracy_loss(model, loader, device):
    ce = nn.CrossEntropyLoss()
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = ce(logits, yb)
        loss_sum += loss.item() * xb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)
    return loss_sum / total, 100.0 * correct / total

def train_one_optimizer(optimizer_name, model_dims, device, epochs=5, batch_size=256,
                        idam_cfg=None, cadam_cfg=None, clip_idam_grad=True):
    train_ds, val_ds = make_synthetic_dataset(in_dim=model_dims[0], n_classes=model_dims[-1])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,  batch_size=batch_size, shuffle=False)

    model = BigMLP(model_dims).to(device)
    ce = nn.CrossEntropyLoss()

    if optimizer_name == "IDAM":
        cfg = dict(lr=0.1, update_interval=1, lr_min=1e-3, lr_max=0.5, beta_disp=0.9, foreach=False)
        if idam_cfg: cfg.update(idam_cfg)
        opt = IDAM(model.parameters(), **cfg)
    elif optimizer_name == "custom_Adam":
        cfg = dict(lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
        if cadam_cfg: cfg.update(cadam_cfg)
        opt = custom_Adam(model.parameters(), **cfg)
    else:
        raise ValueError("optimizer_name must be 'IDAM' or 'custom_Adam'")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        t_ep0 = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            if optimizer_name == "IDAM" and clip_idam_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            ep_loss += loss.item() * xb.size(0)
            ep_correct += (logits.argmax(1) == yb).sum().item()
            ep_total += xb.size(0)

        tr_loss = ep_loss / ep_total
        tr_acc = 100.0 * ep_correct / ep_total
        va_loss, va_acc = eval_accuracy_loss(model, val_loader, device)
        dt = time.time() - t_ep0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["epoch_time"].append(dt)

        print(f"[{optimizer_name}] epoch {ep:02d} | "
              f"train_acc {tr_acc:5.1f}%  val_acc {va_acc:5.1f}%  "
              f"train_loss {tr_loss:.4f}  val_loss {va_loss:.4f}  time {dt:.2f}s")

    total_time = time.time() - t0
    return {
        "final_train_loss": history["train_loss"][-1],
        "final_train_acc":  history["train_acc"][-1],
        "final_val_loss":   history["val_loss"][-1],
        "final_val_acc":    history["val_acc"][-1],
        "avg_epoch_time":   sum(history["epoch_time"]) / len(history["epoch_time"]),
        "total_time":       total_time,
        "history":          history,
    }

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("IDAM vs custom_Adam on a toy classification task")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--foreach", action="store_true", help="Enable foreach fast-path for IDAM")
    parser.add_argument("--idam_lr", type=float, default=0.1)
    parser.add_argument("--idam_lr_min", type=float, default=1e-3)
    parser.add_argument("--idam_lr_max", type=float, default=0.5)
    parser.add_argument("--idam_beta_disp", type=float, default=0.9)
    parser.add_argument("--idam_update_k", type=int, default=1)
    parser.add_argument("--adam_lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} device={device.type}")

    # Model dims for the toy task (input 1024 -> ... -> 10 classes)
    dims = [1024, 2048, 1024, 512, 256, 128, 10]

    idam_cfg = dict(
        lr=args.idam_lr,
        update_interval=args.idam_update_k,
        lr_min=args.idam_lr_min,
        lr_max=args.idam_lr_max,
        beta_disp=args.idam_beta_disp,
        foreach=args.foreach
    )
    cadam_cfg = dict(lr=args.adam_lr)

    print("\n=== Training custom_Adam ===")
    res_cadam = train_one_optimizer(
       	"custom_Adam", dims, device, epochs=args.epochs, batch_size=args.batch_size,
       	idam_cfg=None, cadam_cfg=cadam_cfg, clip_idam_grad=False
    )

    print("\n=== Training IDAM ===")
    res_idam = train_one_optimizer(
        "IDAM", dims, device, epochs=args.epochs, batch_size=args.batch_size,
        idam_cfg=idam_cfg, cadam_cfg=None, clip_idam_grad=True
    )

    print("\n=== SUMMARY (final epoch) ===")
    print(f"IDAM        | train_acc {res_idam['final_train_acc']:.2f}%  val_acc {res_idam['final_val_acc']:.2f}%  "
          f"train_loss {res_idam['final_train_loss']:.4f}  val_loss {res_idam['final_val_loss']:.4f}  "
          f"avg_epoch {res_idam['avg_epoch_time']:.2f}s  total {res_idam['total_time']:.2f}s")
    print(f"custom_Adam | train_acc {res_cadam['final_train_acc']:.2f}%  val_acc {res_cadam['final_val_acc']:.2f}%  "
          f"train_loss {res_cadam['final_train_loss']:.4f}  val_loss {res_cadam['final_val_loss']:.4f}  "
          f"avg_epoch {res_cadam['avg_epoch_time']:.2f}s  total {res_cadam['total_time']:.2f}s")

if __name__ == "__main__":
    main()
