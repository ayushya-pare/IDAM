# Importing Dependencies

import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import Optimizer
from datetime import datetime
import matplotlib.pyplot as plt

# ----------------------------
# Repro
# ----------------------------
torch.manual_seed(42)

# ----------------------------
# VGG16-like Arch and Model
# ----------------------------
arch = [64, 64, 'M',
        128, 128, 'M',
        256, 256, 'M',
        512, 512, 512, 'M',
        512, 512, 512, 'M']  # 5 pools -> 32->1

class VGGNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(arch)
        self.fcs = nn.Sequential(
            nn.Linear(512*1*1, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes)  # logits (use CrossEntropyLoss)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, arch):
        layers = []
        in_channels = self.in_channels
        for x in arch:
            if isinstance(x, int):
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
            else:  # 'M'
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

# ----------------------------
# Custom optimizer (proposed)
# ----------------------------
class proposed_algorithm(Optimizer):
    """
    Step 1: SGD: theta <- theta - lr * grad
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
# Hyperparameters and settings
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 128
EPOCHS = 50
NUM_WORKERS = min(8, os.cpu_count() or 2)
PIN = device == "cuda"

# CIFAR-10 normalization
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
eval_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Datasets
full_train = CIFAR10(root=".", train=True,  transform=train_tfms, download=False)
test_data  = CIFAR10(root=".", train=False, transform=eval_tfms,  download=False)

# Train/Val split (45k/5k)
val_frac = 0.1
n_total  = len(full_train)
n_val    = int(n_total * val_frac)
n_train  = n_total - n_val
train_data, val_data = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))
# Important: no augmentation for val
val_data.dataset.transform = eval_tfms

# Loaders
train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN)
val_loader   = DataLoader(val_data,   batch_size=VAL_BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN)
test_loader  = DataLoader(test_data,  batch_size=VAL_BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN)

# ----------------------------
# Training / Evaluation
# ----------------------------
def train_one_epoch(model, optimizer, criterion, loader):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, criterion, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total

def fit_model(optimizer_ctor, optimizer_kwargs, epochs=EPOCHS, tag="SGD"):
    model = VGGNet(3, 10).to(device)
    optimizer = optimizer_ctor(model.parameters(), **optimizer_kwargs)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [],
               "test_loss": None, "test_accuracy": None}

    os.makedirs("trained_models", exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}/{epochs}", end='\t')
        tr_loss, tr_acc = train_one_epoch(model, optimizer, criterion, train_loader)
        va_loss, va_acc = evaluate(model, criterion, val_loader)
        history["loss"].append(tr_loss); history["accuracy"].append(tr_acc)
        history["val_loss"].append(va_loss); history["val_accuracy"].append(va_acc)
        print(f"Training Loss: {tr_loss:.3f}\tTraining Acc: {tr_acc:.4f}\tVal Loss: {va_loss:.3f}\tVal Acc: {va_acc:.4f}")

        scheduler.step(va_loss)

        if epoch % 5 == 0:
            ts = datetime.now().strftime("%m%d_%H%M%S")
            save_path = os.path.join("trained_models", f'{ts}_epoch{epoch}_{tag}.pth')
            torch.save(model.state_dict(), save_path)

    te_loss, te_acc = evaluate(model, criterion, test_loader)  # final test once
    history["test_loss"] = te_loss
    history["test_accuracy"] = te_acc
    print(f"[{tag}] Final TEST -> loss={te_loss:.4f}, acc={te_acc:.4f}")
    return history

# ----------------------------
# Plotting (joined markers + lines)
# ----------------------------
def plot_performance(histories, titles, savepath="results/benchmark/cifar10_vgg16_results.png"):
    colors  = ['r', 'g', 'b']
    markers = ['o', 's', 'x']
    linestyles = ['-', '-', '-']  # joined
    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    for h, t, c, m, ls in zip(histories, titles, colors, markers, linestyles):
        xs = range(1, len(h["accuracy"]) + 1)
        plt.plot(xs, h["val_accuracy"], marker=m, linestyle=ls, color=c, label=t + ' Val Acc')
        plt.plot(xs, h["accuracy"],     marker=m, linestyle=ls, color=c, alpha=0.5, label=t + ' Train Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    for h, t, c, m, ls in zip(histories, titles, colors, markers, linestyles):
        xs = range(1, len(h["loss"]) + 1)
        plt.plot(xs, h["val_loss"], marker=m, linestyle=ls, color=c, label=t + ' Val Loss')
        plt.plot(xs, h["loss"],     marker=m, linestyle=ls, color=c, alpha=0.5, label=t + ' Train Loss')
    plt.title('Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.tight_layout(); plt.savefig(savepath, format="png", bbox_inches="tight"); plt.show()

# ----------------------------
# Train three runs + Plot
# ----------------------------
if __name__ == "__main__":
    # SGD (with momentum/weight decay as standard for CIFAR-10)
    hist_sgd  = fit_model(torch.optim.SGD,  dict(lr=0.01, momentum=0.9, weight_decay=5e-4), tag="SGD")

    # Adam
    hist_adam = fit_model(torch.optim.Adam, dict(lr=1e-3, weight_decay=5e-4), tag="Adam")

    # Custom
    hist_custom = fit_model(proposed_algorithm, dict(learning_rate=1e-3), tag="Custom")

    # Plot curves
    plot_performance([hist_sgd, hist_adam, hist_custom], ['SGD', 'Adam', 'Custom'])
