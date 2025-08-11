# File: scripts/train_cifar.py
# Description: Trains a ResNet model on CIFAR-100 to benchmark Adam, SGD, and IDAM optimizers.

import os
import time
import argparse
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb

torch.manual_seed(42)

# ========================================
# 1. IDAM
# ========================================
def adaptive_lr_update(eta_prev: float,disp_norm: float, eps: float,lr_min: float = 1e-2, lr_max: float = 1.0) -> float:
    #eta = eta_prev * (1 + gamma * math.exp(-mu * disp_norm * disp_norm))
    eta = eta_prev/(math.sqrt(eps + disp_norm*disp_norm))
    return max(min(eta, lr_max), lr_min)

# ========================================
# 2. Optimized IDAM Optimizer (Scalar adaptive LR per tensor)
# ========================================
class IDAM(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.01, weight_decay: float = 0.0, update_interval: int = 5):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Internal state per parameter-group
        self.update_interval = update_interval
        self.step_counter = 0

        for group in self.param_groups:
            # initialize adaptive LR and previous updates list for each parameter in the group
            group['eta_adaptive'] = group['lr']
            group['prev_updates'] = [torch.zeros_like(p.data) for p in group['params']]

    @torch.no_grad()
    def step(self):
        """
        Perform a single optimization step, fusing gradient updates across each parameter-group.
        """
        self.step_counter += 1

        for group in self.param_groups:
            params = []
            grads = []
            # collect params and grads
            for p in group['params']:
                if p.grad is None:
                    continue
                params.append(p)
                grads.append(p.grad)

            # adapt learning rate every `update_interval` steps
            if self.step_counter % self.update_interval == 0:
                # flatten all previous updates and compute a single displacement norm
                flat = [u.view(-1) for u in group['prev_updates'][:len(grads)]]
                disp_norm = torch.cat(flat).norm().item()
                group['eta_adaptive'] = adaptive_lr_update(
                    group['eta_adaptive'], disp_norm,
                    eps=1e-8, lr_min=1e-2, lr_max=1.0
                )

            eta = group['eta_adaptive']

            # fused compute: updates = -eta * grad for each param
            updates = torch._foreach_mul(grads, -eta)
            torch._foreach_add_(params, updates)

            # save for next displacement calculation
            group['prev_updates'] = [u.clone() for u in updates]    
        return None


# ========================================
# 2b. custom_Adam
# ========================================
class custom_Adam(torch.optim.Optimizer):
    r"""
    Minimal, from-scratch Adam optimizer with optional AMSGrad.
    Matches PyTorch Adam defaults/behavior for:
      - L2 weight decay (NOT decoupled)
      - bias-corrected moment estimates
      - optional AMSGrad variant
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """
        Perform a single optimization step.
        """
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

                # Adam works with dense grads; if sparse, fallback to to_dense()
                if grad.is_sparse:
                    grad = grad.to_dense()

                # L2 weight decay (classic, not decoupled like AdamW)
                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    # State initialization
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # Exponential moving averages of gradient and its square
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt()).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt()).add_(eps)

                # Bias correction
                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t
                step_size = lr * (bias_c2 ** 0.5) / bias_c1

                # Parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return None



# ========================================
# 3. Training and Evaluation
# ========================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = total_loss / len(dataloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = total_loss / len(dataloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

# ========================================
# 4. Main Block
# ========================================
def main():
    parser = argparse.ArgumentParser(description="CIFAR-100 Optimizer Benchmark")
    parser.add_argument("--optimizer", type=str, required=True, choices=["Adam", "SGD", "IDAM","custom_Adam"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--wandb_project", type=str, default="cifar100_training_doover")
    args = parser.parse_args()

    run_name = f"{args.optimizer}_lr={args.lr}"
    wandb.init(project=args.wandb_project, name=run_name, config=args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    model = torchvision.models.resnet18(weights=None, num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.optimizer == 'IDAM':
        optimizer = IDAM(model.parameters(), lr=args.lr)
    elif args.optimizer == 'custom_Adam':
        optimizer = custom_Adam(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)

    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)
        epoch_duration = time.time() - epoch_start_time

        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Duration: {epoch_duration:.2f}s")

        wandb.log({
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "epoch_duration":epoch_duration
        })

    total_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_time:.2f}s")
    wandb.summary['total_training_time_sec'] = total_time
    wandb.finish()

if __name__ == '__main__':
    main()

