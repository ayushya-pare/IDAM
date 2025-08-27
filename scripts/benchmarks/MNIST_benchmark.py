
import os, time
import torch
import numpy as np
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ----------------------------
# Repro
# ----------------------------
torch.manual_seed(42)

# ----------------------------
# Deeper CNN Model (unchanged from your upgrade)
# ----------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
            nn.LogSoftmax(dim=1),  # pair with NLLLoss
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

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
        gamma = 0.1
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
                    p.add_(-ad_lr * g + beta*diff)
        return loss

# ----------------------------
# Data: train/val split + test held out
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST normalization
])

full_train = datasets.MNIST(root="./data", train=True,  download=False, transform=transform)
test_ds    = datasets.MNIST(root="./data", train=False, download=False, transform=transform)

# Create a validation split (e.g., 54k train / 6k val ~ 90/10)
val_frac = 0.1
n_total  = len(full_train)
n_val    = int(n_total * val_frac)
n_train  = n_total - n_val
train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))

batch_size = 256
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
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
    model = CNN().to(device)
    optimizer = optimizer_ctor(model.parameters(), **optimizer_kwargs)
    criterion = nn.NLLLoss()
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "test_loss": None, "test_accuracy": None}
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, optimizer, criterion, train_loader)
        va_loss, va_acc = evaluate(model, criterion, val_loader)   # <-- use VAL, not TEST
        history["loss"].append(tr_loss); history["accuracy"].append(tr_acc)
        history["val_loss"].append(va_loss); history["val_accuracy"].append(va_acc)
        print(f"Epoch {ep:02d}/{epochs} | loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
    # Final hold-out test (only once)
    te_loss, te_acc = evaluate(model, criterion, test_loader)
    history["test_loss"] = te_loss
    history["test_accuracy"] = te_acc
    return history

# ----------------------------
# Visualization
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
    plt.savefig('results/benchmark/mnist_benchmark.png', format="png", bbox_inches="tight")
    plt.show()



# ----------------------------
# Runs: SGD, Adam, Custom + Plot + Final Test
# ----------------------------
if __name__ == "__main__":
    epochs=10
    
    print("\n=== Custom (proposed) ===")
    hist_custom = fit_model(proposed_algorithm, dict(learning_rate=1e-3), epochs=epochs)
    
    print("\n=== SGD ===")
    hist_sgd  = fit_model(optim.SGD,  dict(lr=1e-2), epochs=epochs)

    print("\n=== Adam ===")
    hist_adam = fit_model(optim.Adam, dict(lr=1e-3), epochs=epochs)


    # Plot validation/train curves
    plot_performance_simple([hist_sgd, hist_adam, hist_custom], ['SGD', 'Adam', 'Custom'])

    # Final hold-out test (reported once per model)
    print("\nFinal TEST results:")
    for name, h in zip(['SGD', 'Adam', 'Custom'], [hist_sgd, hist_adam, hist_custom]):
        print(f"{name:6s} -> test_acc={h['test_accuracy']:.4f}, test_loss={h['test_loss']:.4f}")
        plt.show()

    selected_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plot_performance_simple(
        [hist_custom, hist_sgd, hist_adam],
        ['MA-SGD', 'SGD', 'Adam']
        #selected_epochs
    )
