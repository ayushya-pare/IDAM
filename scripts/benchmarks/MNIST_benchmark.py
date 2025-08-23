# PyTorch replication of the provided TensorFlow code (no other changes in intent or flow).
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

        beta = 0.1  # fixed as in the provided TF code
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
                    ad_lr = lr / (diff.abs().pow(2) + 0.1)
                    # new_weights = current - ad_lr * grad + beta * diff
                    update = -ad_lr * grad + beta * diff
                    prev.copy_(current)
                    current.add_(update)

        return loss


class SGDalgorithm(Optimizer):
    """
    Plain SGD with learning_rate (no momentum), to match the provided TF code's SGDalgorithm.
    """
    def __init__(self, params, learning_rate=0.001):
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
        # print(f"Epoch {ep+1:3d}/{epochs}  loss={train_loss:.4f} acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    t1 = time.perf_counter()
    hist.runtime = t1 - t0
    return hist

# ----------------------------
# Train three runs (mirroring TF usage)
# ----------------------------
if __name__ == "__main__":
    # ... all training / plotting code here ...

    # CustomAdagrad
    history_CustomAdagrad = fit_model(CustomAdagrad, dict(learning_rate=1e-4), epochs=5)

    # proposed_algorithm
    history_proposed_algorithm = fit_model(proposed_algorithm, dict(learning_rate=1e-4), epochs=5)

    # SGDalgorithm
    history_SGDalgorithm = fit_model(SGDalgorithm, dict(learning_rate=1e-4), epochs=5)

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
        plt.savefig("myImagePDF.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    selected_epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plot_train_val_metrics(
        [history_proposed_algorithm, history_SGDalgorithm, history_CustomAdagrad],
        ['GE', 'SGD', 'Adam'],
        selected_epochs
    )

    # Reprint results table (as in the original)
    data2 = {
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
    results_table2 = pd.DataFrame(data2)
    print(results_table2)
