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
def adaptive_lr_update(eta_prev: float, disp_norm: float, gamma: float, mu: float, lr_min: float, lr_max: float) -> float:
    eta = eta_prev * (1 + gamma * math.exp(-mu * disp_norm * disp_norm))
    return max(min(eta, lr_max), lr_min)

# ========================================
# 2. Optimized IDAM Optimizer (Scalar adaptive LR per tensor)
# ========================================
class IDAM(torch.optim.Optimizer):
    def __init__(self, params, lr, eps=1e-8, weight_decay=0.0, update_interval=5):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super(IDAM, self).__init__(params, defaults)
        self.step_counter = 0
        self.update_interval = update_interval

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['prev_update'] = torch.zeros_like(p.data)
                state['eta_adaptive'] = lr  # Scalar LR

    def step(self):
        self.step_counter += 1

        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            gamma = 0.001
            mu = 0.1
            lr_min = 1e-2
            lr_max = 1.0

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                eta_prev = state['eta_adaptive']
                displacement = state['prev_update']

                if self.step_counter % self.update_interval == 0:
                    disp_norm = torch.norm(displacement).item()
                    eta = adaptive_lr_update(eta_prev, disp_norm, gamma, mu, lr_min, lr_max)
                    state['eta_adaptive'] = eta

                eta = state['eta_adaptive']
                current_update = -eta * grad
                p.data.add_(current_update)
                state['prev_update'].copy_(current_update)

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
    parser.add_argument("--optimizer", type=str, required=True, choices=["Adam", "SGD", "IDAM"])
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

