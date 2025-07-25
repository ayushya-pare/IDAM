# File: scripts/train_cifar.py
# Description: Trains a ResNet model on CIFAR-100 to benchmark Adam, SGD, and IDAM optimizers.

import os
import time
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb

torch.manual_seed(42)

# ========================================
# 1. Simplified IDAM Optimizer (Only 'original' variant)
# ========================================
class IDAM(torch.optim.Optimizer):
    def __init__(self, params, lr,eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr,eps=eps, weight_decay=weight_decay)
        super(IDAM, self).__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['prev_update'] = torch.zeros_like(p.data)
#                self.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

    def step(self):
        for group in self.param_groups:
#            beta1 = group['beta1']
            lr = group['lr']
            eps = group['eps']
            # Hyperparameters for exponential LR adaptation
            gamma = 0.1     # sensitivity to displacement
            mu = 0.1         # target displacement magnitude
            lr_min = 1e-3
            lr_max = 1.0
            base_lr = group['lr']  # used only on first step


            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                #if group['weight_decay'] > 0.0:
                #    grad = grad.add(p.data, alpha=group['weight_decay'])

                state = self.state[p]
                displacement = state.get('prev_update', torch.zeros_like(p.data))
                #displacement = state['prev_update']


#                eta_adaptive = (lr / torch.sqrt(eps + displacement**2)).clamp(min=1e-2,max=0.1)

#               Logarithmic adaptive learning rate
#                alpha = 100.0  # controls sensitivity of LR decay
 #               eta_adaptive = lr / (torch.log1p(alpha * displacement**2) + eps)
  #              eta_adaptive = eta_adaptive.clamp(min=1e-3, max=1e-1)



                # === Initialize eta_adaptive if not present ===
                if 'eta_adaptive' not in state:
                    state['eta_adaptive'] = torch.full_like(p.data, base_lr)
               
               
               
                eta_prev = state['eta_adaptive']
               
                #eta_adaptive = (eta_prev/(1-displacement**2)).clamp(min=lr_min,max=lr_max)
                eta_adaptive = (eta_prev * (1 - gamma * displacement**2)).clamp(min=lr_min, max=lr_max)

                state['eta_adaptive'] = eta_adaptive
                
                # === Recursive per-element adaptive learning rate ===
                #eta_prev = state['eta_adaptive']
                #update_factor = torch.exp(-gamma * (displacement**2 - mu))
                #eta_adaptive = (eta_prev * update_factor).clamp(min=lr_min, max=lr_max)
                #state['eta_adaptive'] = eta_adaptive 


                # --- Tanh-based adaptive LR update ---
#                eta_prev = state['eta_adaptive']
#                delta = gamma * (displacement**2 - mu)
#                scale = 1 - torch.tanh(delta)
#                eta_adaptive = (eta_prev * scale).clamp(min=lr_min, max=lr_max)
#                state['eta_adaptive'] = eta_adaptive


                # Perform parameter update
                current_update = -eta_adaptive * grad
                state['prev_update'] = current_update
                p.data.add_(current_update)
#               print("adaptive learning rate : ", eta_adaptive)

        return None

# ========================================
# 2. Training and Evaluation
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
# 3. Main Block
# ========================================
def main():
    parser = argparse.ArgumentParser(description="CIFAR-100 Optimizer Benchmark")
    parser.add_argument("--optimizer", type=str, required=True, choices=["Adam", "SGD", "IDAM"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--wandb_project", type=str, default="cifar-100-benchmark")
    args = parser.parse_args()

    run_name = f"{args.optimizer}_lr={args.lr}"
    wandb.init(project=args.wandb_project, name=run_name, config=args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Data Preparation ---
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

    # --- Model, Loss, and Optimizer ---
    model = torchvision.models.resnet18(weights=None, num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.optimizer == 'IDAM':
        optimizer = IDAM(model.parameters(), lr=args.lr)

#    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)
        #scheduler.step()
        epoch_duration = time.time() - epoch_start_time

        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Duration: {epoch_duration:.2f}s")

        wandb.log({
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            #"adaptive_lr": eta_adaptive.mean().item()
            #"learning_rate": scheduler.get_last_lr()[0]
            "epoch time": time.time()-start_time
        })

    total_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_time:.2f}s")
    wandb.summary['total_training_time_sec'] = total_time
    wandb.finish()

if __name__ == '__main__':
    main()
