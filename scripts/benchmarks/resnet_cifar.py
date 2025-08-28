import os, time
import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18  # NEW

# NEW: plotting deps
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Model (now ResNet18 for CIFAR-10, but keeps the same class name to avoid other edits)
# ----------------------------
class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = resnet18(weights=None, num_classes=num_classes)
        #self.logsm = nn.LogSoftmax(dim=1)  # keep NLLLoss compatibility

    def forward(self, x):
        logits = self.backbone(x)
        return logits#self.logsm(logits)

# ----------------------------
# Custom optimizer (proposed)
# ----------------------------
class proposed_algorithm(Optimizer):
    """
    First step: SGD: theta <- theta - lr * grad
    Next: inverse-displacement + momentum
          diff = theta - prev
          ad_lr = lr / (|diff|^2 + 1e-2)
          theta <- theta - ad_lr * grad + beta * diff
    """
    def __init__(self, params, learning_rate=1e-3):
        super().__init__(params, dict(lr=learning_rate))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        beta = 0.1
        gamma = 0.02
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "initialized" not in state:
                    state["initialized"] = True
                    state["prev"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    p.add_(g, alpha=-lr)           # first: plain SGD
                    state["prev"].copy_(p)
                else:
                    prev = state["prev"]
                    diff = p - prev
                    ad_lr = lr / (diff.abs().pow(2) + gamma)
                    prev.copy_(p)
                    p.add_(-ad_lr * g + beta * diff)
        return loss

# ----------------------------
# Data
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# CIFAR-10 normalization
cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std  = (0.2470, 0.2435, 0.2616)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std),
])

train_ds = datasets.CIFAR10(root="./data", train=True,  download=False, transform=transform)
test_ds  = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)

batch_size = 256
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

# ----------------------------
# Train / Eval helpers
# ----------------------------
def train_one_epoch(model, optimizer, criterion, loader):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, criterion, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

def fit_model(optimizer_ctor, optimizer_kwargs, epochs=10):
    model = MLP().to(device)  # still called MLP, now a ResNet under the hood
    optimizer = optimizer_ctor(model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, optimizer, criterion, train_loader)
        va_loss, va_acc = evaluate(model, criterion, test_loader)
        history["loss"].append(tr_loss); history["accuracy"].append(tr_acc)
        history["val_loss"].append(va_loss); history["val_accuracy"].append(va_acc)
        print(f"Epoch {ep:02d}/{epochs} | loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
    return history

# ----------------------------
# Plotting helper (unchanged style; file name updated)
# ----------------------------
def plot_performance_simple(histories, titles):
    # Style (matches your reference)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    font_title  = {'family': 'serif', 'color': 'black',   'weight': 'normal', 'size': 16}
    font_labels = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 18}
    font_legend = {'size': 16}

    # X axis (all epochs, using first history length)
    n = len(histories[0]['loss'])
    ep = np.arange(n)  # 0-based for plotting

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.set_xlabel('Epoch', fontdict=font_labels)
    ax1.set_ylabel('Loss',  fontdict=font_labels, color='tab:red')
    ax2.set_xlabel('Epoch', fontdict=font_labels)
    ax2.set_ylabel('Accuracy', fontdict=font_labels, color='tab:blue')

    for h, title, color in zip(histories, titles, colors):
        # Loss panel
        ax1.plot(ep, np.array(h['loss']),     'o-', color=color, label=f'{title} Train Loss', markersize=8)
        ax1.plot(ep, np.array(h['val_loss']), 's--', color=color, label=f'{title} Val  Loss', markersize=8)
        # Accuracy panel
        ax2.plot(ep, np.array(h['accuracy']),     'o-', color=color, label=f'{title} Train Accuracy', markersize=8)
        ax2.plot(ep, np.array(h['val_accuracy']), 's--', color=color, label=f'{title} Val  Accuracy', markersize=8)

    ax1.legend(loc='upper right', prop=font_legend)
    ax2.legend(loc='lower right', prop=font_legend)
    ax1.title.set_text('Loss Comparison')
    ax2.title.set_text('Accuracy Comparison')

    plt.tight_layout()
    os.makedirs('results/benchmark', exist_ok=True)
    plt.savefig('results/benchmark/cifar10_resnet.png', format="png", bbox_inches="tight")
    plt.show()

# ----------------------------
# Runs: SGD, Adam, Custom
# ----------------------------
if __name__ == "__main__":
    
    print("\n=== Custom (proposed) ===")
    hist_custom = fit_model(proposed_algorithm, dict(learning_rate=1e-3), epochs=10)
    
    print("\n=== SGD ===")
    hist_sgd  = fit_model(optim.SGD,  dict(lr=1e-2), epochs=10)

    print("\n=== Adam ===")
    hist_adam = fit_model(optim.Adam, dict(lr=1e-3), epochs=10)


    # Tiny results summary
    def last(h, key): return h[key][-1]
    print("\nFinal validation:")
    print(f"Custom -> acc={last(hist_custom,'val_accuracy'):.4f}, loss={last(hist_custom,'val_loss'):.4f}")
    print(f"SGD    -> acc={last(hist_sgd,'val_accuracy'):.4f}, loss={last(hist_sgd,'val_loss'):.4f}")
    print(f"Adam   -> acc={last(hist_adam,'val_accuracy'):.4f}, loss={last(hist_adam,'val_loss'):.4f}")

    # ----------------------------
    # Plot all three runs
    # ----------------------------
    plot_performance_simple(
        histories=[hist_sgd, hist_adam, hist_custom],
        titles=["SGD", "Adam", "Custom"]
    )
