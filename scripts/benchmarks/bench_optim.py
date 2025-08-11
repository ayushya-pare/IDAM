import time, math, torch, torch.nn as nn
from statistics import mean, median

torch.manual_seed(0)

def adaptive_lr_update(eta_prev, disp_norm, eps, lr_min=1e-2, lr_max=1.0):
    eta = eta_prev/(math.sqrt(eps + disp_norm*disp_norm))
    return max(min(eta, lr_max), lr_min)

# ========================================
# IDAM (per-parameter loop, no foreach)
# ========================================
class IDAM(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.01, weight_decay: float = 0.0, update_interval: int = 5):
        # weight_decay kept for API parity; not applied (matches your original)
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
            params_with_grad, grads, idxs = [], [], []

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    g = g.to_dense()
                params_with_grad.append(p)
                grads.append(g)
                idxs.append(i)

            if self.step_counter % self.update_interval == 0 and idxs:
                flat_prev = [group['prev_updates'][i].view(-1) for i in idxs]
                disp_norm = torch.cat(flat_prev).norm().item() if flat_prev else 0.0
                group['eta_adaptive'] = adaptive_lr_update(
                    group['eta_adaptive'], disp_norm, eps=1e-8, lr_min=1e-2, lr_max=1.0
                )

            eta = group['eta_adaptive']

            for p, g, i in zip(params_with_grad, grads, idxs):
                p.add_(g, alpha=-eta)
                group['prev_updates'][i] = (-eta * g).clone()

        return None

# ========================================
# custom_Adam (from-scratch Adam)
# ========================================
class custom_Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0, amsgrad=False):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad))
    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            lr, (b1,b2), eps, wd, ams = g['lr'], g['betas'], g['eps'], g['weight_decay'], g['amsgrad']
            for p in g['params']:
                if p.grad is None: continue
                grad = p.grad.to_dense() if p.grad.is_sparse else p.grad
                if wd: grad = grad.add(p, alpha=wd)

                s = self.state[p]
                if len(s)==0:
                    s['step']=0
                    s['exp_avg']=torch.zeros_like(p, memory_format=torch.preserve_format)
                    s['exp_avg_sq']=torch.zeros_like(p, memory_format=torch.preserve_format)
                    if ams: s['max_exp_avg_sq']=torch.zeros_like(p, memory_format=torch.preserve_format)

                s['step']+=1; t=s['step']
                s['exp_avg'].mul_(b1).add_(grad, alpha=1-b1)
                s['exp_avg_sq'].mul_(b2).addcmul_(grad, grad, value=1-b2)

                denom = (s['max_exp_avg_sq'] if ams else s['exp_avg_sq'])
                if ams: torch.maximum(denom, s['exp_avg_sq'], out=denom)
                denom = denom.sqrt().add_(eps)

                step_size = lr * ((1-b2**t)**0.5) / (1-b1**t)
                p.addcdiv_(s['exp_avg'], denom, value=-step_size)
        return None

# --- synthetic model & timing ---
class BigMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers=[]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1], bias=True))
            if i < len(dims)-2: layers.append(nn.ReLU())
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

def set_random_grads(model):
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)

def trimmed_mean(xs, lo=0.05, hi=0.95):
    if not xs: return float('nan')
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    lo_i = int(n*lo)
    hi_i = int(n*hi)
    if hi_i <= lo_i:  # fallback if too few samples
        return mean(xs_sorted)
    return mean(xs_sorted[lo_i:hi_i])

def time_steps(opt, model, device, warmup=10, iters=200):
    # Warmup
    for _ in range(warmup):
        set_random_grads(model)
        if device.type == "cuda": torch.cuda.synchronize()
        opt.step()
        if device.type == "cuda": torch.cuda.synchronize()

    times=[]
    for _ in range(iters):
        set_random_grads(model)
        if device.type == "cuda": torch.cuda.synchronize()
        t0=time.perf_counter()
        opt.step()
        if device.type == "cuda": torch.cuda.synchronize()
        t1=time.perf_counter()
        times.append(t1-t0)

    return {
        "mean": mean(times),
        "median": median(times),
        "trimmed_mean_5_95": trimmed_mean(times, 0.05, 0.95),
        "min": min(times),
        "max": max(times),
        "n": len(times),
    }

# ---- run ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dims=[1024,2048,2048,1024,512,256,128,10]

# Two separate copies per optimizer/setting to avoid shared state
m_idam_1  = BigMLP(dims).to(device)
m_idam_10 = BigMLP(dims).to(device)
m_adam    = BigMLP(dims).to(device)

idam_1  = IDAM(m_idam_1.parameters(),  lr=0.01, update_interval=1)   # worst-case overhead
idam_10 = IDAM(m_idam_10.parameters(), lr=0.01, update_interval=10)  # lighter overhead
cadam   = custom_Adam(m_adam.parameters(), lr=1e-3, weight_decay=0.0, amsgrad=False)  # weight_decay=0.0

print(f"torch {torch.__version__} device={device.type}")

stats_idam1  = time_steps(idam_1,  m_idam_1,  device)
stats_idam10 = time_steps(idam_10, m_idam_10, device)
stats_cadam  = time_steps(cadam,   m_adam,    device)

def pretty(name, s):
    print(f"{name:18s} mean={s['mean']:.6f}s  median={s['median']:.6f}s  trimmed(5–95%)={s['trimmed_mean_5_95']:.6f}s  min={s['min']:.6f}s  max={s['max']:.6f}s  n={s['n']}")

pretty("IDAM (k=1)",  stats_idam1)
pretty("IDAM (k=10)", stats_idam10)
pretty("custom_Adam", stats_cadam)
