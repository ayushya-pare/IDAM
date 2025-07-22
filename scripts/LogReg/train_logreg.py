# File: scripts/train_logreg.py
# Description: Trains a Logistic Regression model to benchmark advanced IDAM variants.

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
# 1. Advanced IDAM Optimizer with Variants
# ========================================
class IDAM(torch.optim.Optimizer):
    """
    An advanced IDAM optimizer with multiple configurable variants.
    """
    def __init__(self, params, lr=0.1, eps=1e-8, weight_decay=0.0, beta1=0.9, beta2=0.999, variant='original'):
        if variant not in ['original', 'linear', 'exp', 'log', 'v']:
            raise ValueError(f"Invalid IDAM variant: {variant}")
            
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, beta1=beta1, beta2=beta2, variant=variant)
        super(IDAM, self).__init__(params, defaults)
        
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['prev_update'] = torch.zeros_like(p.data)
                self.state[p]['momentum_buffer'] = torch.zeros_like(p.data)
                if group['variant'] == 'v':
                    self.state[p]['displacement_variance'] = torch.zeros_like(p.data)

    def step(self):
        for group in self.param_groups:
            beta1 = group['beta1']
            beta2 = group['beta2']
            variant = group['variant']
            lr = group['lr']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if group['weight_decay'] > 0.0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                state = self.state[p]
                displacement = state['prev_update']
                
                # --- Variant-specific adaptive learning rate calculation ---
                if variant == 'v':
                    # IDAM-V with second-order displacement variance
                    disp_var = state['displacement_variance']
                    disp_var.mul_(beta2).addcmul_(displacement, displacement, value=1 - beta2)
                    eta_adaptive = lr / (disp_var.sqrt() + eps)
                else:
                    # All other variants are based on direct displacement
                    if variant == 'original':
                        eta_adaptive = lr / torch.sqrt(1 + displacement**2 + eps)
                    elif variant == 'linear':
                        eta_adaptive = lr / (1 + displacement.abs() + eps)
                    elif variant == 'exp':
                        eta_adaptive = lr * torch.exp(-displacement.abs())
                    elif variant == 'log':
                        # **FIXED LINE**: Added 1 to the denominator to prevent division by zero.
                        eta_adaptive = lr / (1 + torch.log(1 + displacement**2 + eps))

                current_update = -eta_adaptive * grad
                state['prev_update'] = current_update

                # Update the momentum buffer
                momentum_buffer = state['momentum_buffer']
                momentum_buffer.mul_(beta1).add_(current_update)

                # Apply the momentum-smoothed update
                p.data.add_(momentum_buffer)
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
    parser.add_argument("--wandb_project", type=str, default="idam-variants-benchmark")
    # Add arguments for IDAM variants
    parser.add_argument("--idam_variant", type=str, default="original", help="Variant of IDAM to use.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for IDAM-V variant.")
    args = parser.parse_args()

    run_name = f"{args.optimizer}_lr={args.lr}"
    if args.optimizer == "IDAM":
        run_name = f"IDAM_{args.idam_variant}_lr={args.lr}"

    wandb.init(project=args.wandb_project, name=run_name, config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X, y = make_classification(n_samples=10000, n_features=100, n_informative=20, n_redundant=30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = LogisticRegression(n_features=100).to(device)
    criterion = nn.BCELoss()
    
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "IDAM":
        optimizer = IDAM(model.parameters(), lr=args.lr, variant=args.idam_variant, beta2=args.beta2)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        wandb.log({"train_loss": train_loss, "train_accuracy": train_acc, "val_accuracy": test_acc})

    total_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_time:.2f}s")
    wandb.summary['total_training_time_sec'] = total_time
    
    wandb.finish()

if __name__ == "__main__":
    main()

