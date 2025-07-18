# File: scripts/train_lora.py
# Description: A unified script with a FASTER IDAM implementation and early stopping.

import os
import time
import argparse
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# ========================================
# 1. OPTIMIZED IDAM Optimizer 
# ========================================
class IDAM(torch.optim.Optimizer):
    """
    An optimized implementation of the IDAM optimizer.
    This version avoids the expensive .clone() operation by storing the previous
    update step instead of the entire previous parameter tensor.
    """
    def __init__(self, params, lr=0.1, eps=1e-8):
        defaults = dict(lr=lr, eps=eps)
        super(IDAM, self).__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                # Initialize a state to store the previous update, which acts as the displacement
                self.state[p]['prev_update'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]

                # The displacement for this step is the update we applied in the *previous* step.
                displacement = state['prev_update']
                
                # Adapt the learning rate based on the previous displacement
                eta_adaptive = group['lr'] / torch.sqrt(1 + displacement**2 + group['eps'])
                
                # Calculate the update for the CURRENT step
                current_update = -eta_adaptive * grad
                
                # Store the current update for the NEXT step's displacement calculation.
                # This is far more efficient than cloning the entire parameter tensor.
                state['prev_update'] = current_update
                
                # Apply the update in-place
                p.data.add_(current_update)
        
        return loss

# ========================================
# 2. Training and Evaluation (No changes needed here)
# ========================================

def train_epoch(model, dataloader, optimizer, lr_scheduler, device, scaler):
    model.train()
    total_loss = 0
    epoch_start_time = time.time()
    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()
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
# 3. Main Block (No changes needed here)
# ========================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer with LoRA.")
    parser.add_argument("--optimizer", type=str, required=True, choices=["AdamW", "IDAM"], help="Optimizer to use.")
    parser.add_argument("--epochs", type=int, default=3, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW or base LR (alpha) for IDAM.")
    parser.add_argument("--patience", type=int, default=1, help="Number of epochs to wait for improvement before stopping.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "distilbert-base-uncased"
    dataset_name = "glue"
    subset_name = "sst2"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(os.environ.get("WORK", "."), "hf_cache")

    print(f"--- Starting Experiment: Optimizer={args.optimizer}, LR/Alpha={args.lr} ---")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw_datasets = load_dataset(dataset_name, subset_name, cache_dir=cache_dir)

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=args.batch_size)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q_lin", "v_lin"]
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()
    
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == "IDAM":
        optimizer = IDAM(model.parameters(), lr=args.lr)
    
    num_training_steps = args.epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    scaler = GradScaler()

    results = []
    total_training_time = 0
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    adapter_path = os.path.join(output_dir, f"adapter_{args.optimizer}")

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss, epoch_duration = train_epoch(model, train_dataloader, optimizer, lr_scheduler, device, scaler)
        val_accuracy = evaluate(model, eval_dataloader, device)
        
        total_training_time += epoch_duration
        
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Accuracy = {val_accuracy:.4f}, Duration = {epoch_duration:.2f}s")
        results.append({
            "epoch": epoch + 1, "train_loss": train_loss, "val_accuracy": val_accuracy, "epoch_duration_sec": epoch_duration
        })

        if val_accuracy > best_val_accuracy:
            print(f"Validation accuracy improved from {best_val_accuracy:.4f} to {val_accuracy:.4f}. Saving model.")
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            model.save_pretrained(adapter_path)
        else:
            epochs_no_improve += 1
            print(f"Validation accuracy did not improve. Patience: {epochs_no_improve}/{args.patience}")

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {args.patience} epochs with no improvement.")
            break

    results_df = pd.DataFrame(results)
    summary = pd.DataFrame([{"epoch": "total", "epoch_duration_sec": total_training_time}])
    results_df = pd.concat([results_df, summary], ignore_index=True)

    output_filename = os.path.join(output_dir, f"results_{args.optimizer}.csv")
    results_df.to_csv(output_filename, index=False)
    print(f"\n--- Experiment Complete ---")
    print(f"Saved final results to {output_filename}")
    print(f"Best adapter for {args.optimizer} saved to {adapter_path}")

if __name__ == "__main__":
    main()
