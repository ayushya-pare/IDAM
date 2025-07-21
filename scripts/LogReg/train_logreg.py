# File: scripts/train_logreg.py
# Description: Trains a Logistic Regression model on a synthetic dataset to benchmark optimizers.

import os
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import wandb

# ========================================
# 1. IDAM Optimizer (Best Performing Version)
# ========================================
class IDAM(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super(IDAM, self).__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['prev_update'] = torch.zeros_like(p.data)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if group['weight_decay'] > 0.0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                state = self.state[p]
                displacement = state['prev_update']
                eta_adaptive = group['lr'] / torch.sqrt(1 + displacement**2 + group['eps'])
                current_update = -eta_adaptive * grad
                state['prev_update'] = current_update
                p.data.add_(current_update)
        return None

# ========================================
# 2. Logistic Regression Model
# ========================================
class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# ========================================
# 3. Training and Evaluation
# ========================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for features, target in dataloader:
        features, target = features.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, target.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for features, target in dataloader:
            features, target = features.to(device), target.to(device)
            output = model(features)
            pred = (output > 0.5).float()
            correct += (pred == target.unsqueeze(1).float()).sum().item()
    return correct / len(dataloader.dataset)

# ========================================
# 4. Main Block
# ========================================
def main():
    parser = argparse.ArgumentParser(description="Logistic Regression Optimizer Benchmark")
    parser.add_argument("--optimizer", type=str, required=True, choices=["Adam", "SGD", "IDAM"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--wandb_project", type=str, default="idam-logreg-benchmark")
    args = parser.parse_args()

    # --- Wandb Initialization ---
    wandb.init(
        project=args.wandb_project,
        name=f"{args.optimizer}_lr={args.lr}",
        config=args
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Data Generation ---
    X, y = make_classification(n_samples=10000, n_features=100, n_informative=20, n_redundant=30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = LogisticRegression(n_features=100).to(device)
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss for logistic regression
    
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "IDAM":
        optimizer = IDAM(model.parameters(), lr=args.lr)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        epoch_duration = time.time() - epoch_start_time
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}, Duration: {epoch_duration:.2f}s")
        
        # --- Wandb Logging ---
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": test_acc,
            "epoch_duration_sec": epoch_duration
        })

    total_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_time:.2f}s")
    wandb.summary['total_training_time_sec'] = total_time
    
    wandb.finish()

if __name__ == "__main__":
    main()

