# File: scripts/train_lora.py
# Description: An advanced, reproducible script to compare AdamW and IDAM, incorporating best practices.

import os
import time
import argparse
import pandas as pd
import numpy as np
import random
import torch
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast
from torch.cuda.amp import GradScaler

# ========================================
# 0. Reproducibility Helper
# ========================================
def set_seed(seed):
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========================================
# 1. GRADIENT-INFORMED IDAM Optimizer 
# ========================================
class IDAM(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super(IDAM, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('IDAM does not support sparse gradients')
                
                # Apply weight decay
                if group['weight_decay'] > 0.0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Use the gradient itself as the anticipated displacement
                displacement = grad
                
                # Adapt the learning rate based on the gradient's magnitude
                eta_adaptive = group['lr'] / torch.sqrt(1 + displacement**2 + group['eps'])
                
                # Apply the update
                p.data.add_(-eta_adaptive * grad)
        return None

# ========================================
# 2. Training and Evaluation
# ========================================

def train_epoch(model, dataloader, optimizer, lr_scheduler, device, scaler, grad_clip):
    model.train()
    total_loss = 0
    epoch_start_time = time.time()
    for batch in tqdm(dataloader, desc="Training"):
        # Update learning rate for each step
        lr = lr_scheduler.get_last_lr()[0]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(device_type=device.type):
            outputs = model(**batch)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        
        # Gradient Clipping
        if grad_clip > 0.0:
            scaler.unscale_(optimizer) # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item()
    
    epoch_duration = time.time() - epoch_start_time
    avg_loss = total_loss / len(dataloader)
    return avg_loss, epoch_duration

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    return accuracy_score(all_labels, all_preds)

# ========================================
# 3. Main Block
# ========================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer with LoRA.")
    parser.add_argument("--optimizer", type=str, required=True, choices=["AdamW", "IDAM"], help="Optimizer to use.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Max learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="Number of warmup epochs.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value (0.0 for no clipping).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "distilbert-base-uncased"
    dataset_name = "glue"
    subset_name = "sst2"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(os.environ.get("WORK", "."), "hf_cache")

    print(f"--- Starting Experiment: Optimizer={args.optimizer}, LR={args.lr}, Seed={args.seed} ---")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw_datasets = load_dataset(dataset_name, subset_name, cache_dir=cache_dir)

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["sentence", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=args.batch_size)

    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q_lin", "v_lin"])
    # Removed attn_implementation="flash_attention_2" to resolve the ImportError
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = get_peft_model(model, lora_config)
    model.to(device)
    
    # --- Advanced Optimizer Setup ---
    # Create parameter groups to apply weight decay only to non-bias/LayerNorm weights
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, fused=True)
    elif args.optimizer == "IDAM":
        optimizer = IDAM(optimizer_grouped_parameters, lr=args.lr)
    
    # --- Cosine Learning Rate Scheduler ---
    num_training_steps = args.epochs * len(train_dataloader)
    num_warmup_steps = args.warmup_epochs * len(train_dataloader)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps - num_warmup_steps)

    scaler = GradScaler()

    # --- Training Loop ---
    results = []
    total_training_time = 0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss, epoch_duration = train_epoch(model, train_dataloader, optimizer, lr_scheduler, device, scaler, args.grad_clip)
        val_accuracy = evaluate(model, eval_dataloader, device)
        total_training_time += epoch_duration
        
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Accuracy = {val_accuracy:.4f}, Duration = {epoch_duration:.2f}s")
        results.append({"epoch": epoch + 1, "train_loss": train_loss, "val_accuracy": val_accuracy, "epoch_duration_sec": epoch_duration})

    # --- Save Results ---
    results_df = pd.DataFrame(results)
    output_filename = os.path.join(output_dir, f"results_{args.optimizer}_lr{args.lr}_seed{args.seed}.csv")
    results_df.to_csv(output_filename, index=False)
    print(f"\n--- Experiment Complete ---")
    print(f"Saved final results to {output_filename}")

if __name__ == "__main__":
    main()
