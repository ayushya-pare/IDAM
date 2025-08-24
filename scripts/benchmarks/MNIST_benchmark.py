# - CustomAdagrad optimizer
# - proposed_algorithm (inverse-displacement + momentum)
# - SGDalgorithm
# - MLP model on MNIST (64 ReLU -> 10, trained for 100 epochs)
# - Training histories collected to mimic Keras History objects
# - Plots and results table reproduced

import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----------------------------
# Utilities: History container
# ----------------------------
class History:
    def __init__(self):
        self.history = {
            "accuracy": [],
            "val_accuracy": [],
            "loss": [],
            "val_loss": [],
        }

# ----------------------------
# Custom Optimizers
# ----------------------------

class CustomAdagrad(Optimizer):
    """
    Adagrad: accumulator += grad^2 ; param -= lr * grad / (sqrt(accumulator) + eps)
    """
    def __init__(self, params, learning_rate=0.001, epsilon=1e-7):
        defaults = dict(lr=learning_rate, eps=epsilon)
        super(CustomAdagrad, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "accumulator" not in state:
                    state["accumulator"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                acc = state["accumulator"]
                acc.add_(grad * grad)
                denom = acc.sqrt().add_(eps)
                p.addcdiv_(grad, denom, value=-lr)
        return loss


class proposed_algorithm(Optimizer):
    """
    First step: plain SGD: theta <- theta - lr * grad
    Next steps: inverse-displacement + momentum:
        diff = theta - prev
        ad_lr = lr / (|diff|^2 + 0.1)
        theta <- theta - ad_lr * grad + beta * diff
    Uses beta = 0.1 (fixed, as in the TF code).
    """
    def __init__(self, params, learning_rate=0.001):
        defaults = dict(lr=learning_rate)
        super(proposed_algorithm, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        beta = 0.9  # fixed as in the provided TF code
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # Initialize state buffers
                if "initialized" not in state:
                    state["initialized"] = False
                    state["previous_weights"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                prev = state["previous_weights"]
                current = p

                if not state["initialized"]:
                    # First step: plain SGD
                    current.add_(grad, alpha=-lr)
                    prev.copy_(current)
                    state["initialized"] = True
                else:
                    diff = current - prev
                    # ad_lr is element-wise
                    ad_lr = lr / (diff.abs().pow(2) + 1e-2)
                    # new_weights = current - ad_lr * grad + beta * diff
                    update = -ad_lr * grad + beta * diff
                    prev.copy_(current)
                    current.add_(update)

        return loss


class SGDalgorithm(Optimizer):
    """
    Plain SGD with learning_rate (no momentum), to match the provided TF code's SGDalgorithm.
    """
    def __init__(self, params, learning_rate=0.01):
        defaults = dict(lr=learning_rate)
        super(SGDalgorithm, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(p.grad, alpha=-lr)
        return loss

# ----------------------------
# Model (replicates: Dense(64, relu) -> Dense(10), with log-softmax for NLLLoss)
# ----------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
        self.logsm = nn.LogSoftmax(dim=1)  # To pair with NLLLoss (equivalent to softmax + cross-entropy)

    def forward(self, x):
        # x: (N, 1, 28, 28) -> flatten to (N, 784)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.logsm(x)

# ----------------------------
# Data
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : ", device)

transform = transforms.ToTensor()
train_ds = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=False, transform=transform)

# Keras default batch_size is 32 if unspecified; we mirror that.
batch_size = 256
num_workers = min(4, os.cpu_count() or 4) if device.type != "cuda" else 0

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=(device.type == "cuda"),
    persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
)
test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=(device.type == "cuda"),
    persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
)

#x, y = x.to(device, non_blocking=(device.type == "cuda")), y.to(device, non_blocking=(device.type == "cuda"))


#train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
#test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

# ----------------------------
# Training / Eval
# ----------------------------
def train_one_epoch(model, optimizer, criterion, loader):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)                 # log-probs
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, criterion, loader):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total

def fit_model(optimizer_ctor, optimizer_kwargs, epochs=100):
    model = MLP().to(device)
    optimizer = optimizer_ctor(model.parameters(), **optimizer_kwargs)
    criterion = nn.NLLLoss()  # pairs with model's LogSoftmax
    hist = History()
    t0 = time.perf_counter()
    for ep in range(epochs):
        train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader)
        val_loss, val_acc     = evaluate(model, criterion, test_loader)
        hist.history["loss"].append(train_loss)
        hist.history["accuracy"].append(train_acc)
        hist.history["val_loss"].append(val_loss)
        hist.history["val_accuracy"].append(val_acc)
        print(f"Epoch {ep+1:3d}/{epochs}  loss={train_loss:.4f} acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    t1 = time.perf_counter()
    hist.runtime = t1 - t0
    return hist

# ----------------------------
# Train three runs (mirroring TF usage)
# ----------------------------
if __name__ == "__main__":
    # ... all training / plotting code here ...

    # CustomAdagrad
    history_CustomAdagrad = fit_model(torch.optim.Adam, dict(learning_rate=1e-3), epochs=50)

    # proposed_algorithm
    history_proposed_algorithm = fit_model(proposed_algorithm, dict(learning_rate=1e-3), epochs=50)

    # SGDalgorithm
    history_SGDalgorithm = fit_model(torch.optim.SGD, dict(learning_rate=1e-2), epochs=50)

    # ----------------------------
    # Plotting (replicated from the TF code structure)
    # ----------------------------
    def plot_performance(histories, titles):
        colors = ['r', 'g', 'b']
        markers = ['o', '--', 'x']
        plt.figure(figsize=(14, 5))

        # Accuracy
        plt.subplot(1, 2, 1)
        for history, title, color, marker in zip(histories, titles, colors, markers):
            plt.plot(history.history['val_accuracy'], marker + color, label=title + ' Val Accuracy')
            plt.plot(history.history['accuracy'],     marker + color, label=title + ' Train Accuracy', alpha=0.5)
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss
        plt.subplot(1, 2, 2)
        for history, title, color, marker in zip(histories, titles, colors, markers):
            plt.plot(history.history['val_loss'], marker + color, label=title + ' Val Loss')
            plt.plot(history.history['loss'],     marker + color, label=title + ' Train Loss', alpha=0.5)
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    # Train and evaluate (names kept identical to the original snippet)
    hist_custom = history_proposed_algorithm
    hist_sgd    = history_SGDalgorithm
    hist_adam   = history_CustomAdagrad

    plot_performance([hist_custom, hist_sgd, hist_adam], ['Custom Optimizer', 'SGD', 'Adam'])

    # Final results table (same labels as original)
    data = {
        'Optimizer': ['Custom Optimizer', 'SGD', 'Adam'],
        'Validation Accuracy': [
            hist_custom.history['val_accuracy'][-1],
            hist_sgd.history['val_accuracy'][-1],
            hist_adam.history['val_accuracy'][-1]
        ],
        'Validation Loss': [
            hist_custom.history['val_loss'][-1],
            hist_sgd.history['val_loss'][-1],
            hist_adam.history['val_loss'][-1]
        ]
    }
    results_table = pd.DataFrame(data)
    print(results_table)

    # ----------------------------
    # Additional plot (as in the original)
    # ----------------------------
    def plot_train_val_metrics(histories, titles, epochs):
        # Define colors and font sizes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Dark Blue, Dark Orange, Dark Green
        font_title = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 16}
        font_labels = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 18}
        font_legend = {'size': 16}

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.set_xlabel('Epoch', fontdict=font_labels)
        ax1.set_ylabel('Loss', fontdict=font_labels)
        ax2.set_xlabel('Epoch', fontdict=font_labels)
        ax2.set_ylabel('Accuracy', fontdict=font_labels)

        epoch_indices = [ep - 1 for ep in epochs]  # zero-based

        for history, title, color in zip(histories, titles, colors):
            selected_train_loss = np.array(history.history['loss'])[epoch_indices]
            selected_val_loss   = np.array(history.history['val_loss'])[epoch_indices]
            selected_train_acc  = np.array(history.history['accuracy'])[epoch_indices]
            selected_val_acc    = np.array(history.history['val_accuracy'])[epoch_indices]
            selected_epochs     = np.array(epochs) - 1

            ax1.plot(selected_epochs, selected_train_loss, 'o-', color=color, label=f'{title} Train Loss', markersize=8)
            ax1.plot(selected_epochs, selected_val_loss,   's--', color=color, label=f'{title} Val Loss', markersize=8)
            ax2.plot(selected_epochs, selected_train_acc,  'o-', color=color, label=f'{title} Train Accuracy', markersize=8)
            ax2.plot(selected_epochs, selected_val_acc,    's--', color=color, label=f'{title} Val Accuracy', markersize=8)

        ax1.legend(loc='upper right', prop=font_legend)
        ax2.legend(loc='lower right', prop=font_legend)
        ax1.title.set_text('Loss Comparison')
        ax2.title.set_text('Accuracy Comparison')

        plt.tight_layout()
import os, time
import torch
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
        beta = 0.9
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
                    ad_lr = lr / (diff.abs().pow(2) + 1e-2)
                    prev.copy_(p)
                    p.add_(-ad_lr * g + beta * diff)
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
def plot_performance(histories, titles):
    colors  = ['r', 'g', 'b']
    markers = ['o', '--', 'x']
    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    for h, title, color, marker in zip(histories, titles, colors, markers):
        plt.plot(h["val_accuracy"], marker + color, label=title + ' Val Accuracy')
        plt.plot(h["accuracy"],     marker + color, label=title + ' Train Accuracy', alpha=0.5)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    for h, title, color, marker in zip(histories, titles, colors, markers):
        plt.plot(h["val_loss"], marker + color, label=title + ' Val Loss')
        plt.plot(h["loss"],     marker + color, label=title + ' Train Loss', alpha=0.5)
    plt.title('Model Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout(); plt.show()

# ----------------------------
# Runs: SGD, Adam, Custom + Plot + Final Test
# ----------------------------
if __name__ == "__main__":
    print("\n=== SGD ===")
    hist_sgd  = fit_model(optim.SGD,  dict(lr=1e-2), epochs=50)

    print("\n=== Adam ===")
    hist_adam = fit_model(optim.Adam, dict(lr=1e-3), epochs=50)

    print("\n=== Custom (proposed) ===")
    hist_custom = fit_model(proposed_algorithm, dict(lr=1e-3), epochs=50)

    # Plot validation/train curves
    plot_performance([hist_sgd, hist_adam, hist_custom], ['SGD', 'Adam', 'Custom'])

    # Final hold-out test (reported once per model)
    print("\nFinal TEST results (no peeking during training):")
    for name, h in zip(['SGD', 'Adam', 'Custom'], [hist_sgd, hist_adam, hist_custom]):
        print(f"{name:6s} -> test_acc={h['test_accuracy']:.4f}, test_loss={h['test_loss']:.4f}")
        plt.show()

    selected_epochs = [10, 20, 30, 40, 50]#, 60, 70, 80, 90, 100]
    plot_train_val_metrics(
        [history_proposed_algorithm, history_SGDalgorithm, history_CustomAdagrad],
        ['GE', 'SGD', 'Adam'],
        selected_epochs
    )
