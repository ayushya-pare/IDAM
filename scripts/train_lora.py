# File: scripts/train_comparison.py
# Description: A unified script to fine-tune DistilBERT on SST-2 using either AdamW or IDAM.

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
# 1. IDAM Optimizer 
# ========================================
class IDAM(torch.optim.Optimizer):
    """
    Implements the Inverse Displacement(IDAM) optimizer.
    """
    def __init__(self, params, lr=0.1, eps=1e-8):
        defaults = dict(lr=lr, eps=eps)
        super(IDAM, self).__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                # Initialize a state to store the parameter's previous value
                self.state[p]['prev_param'] = torch.zeros_like(p.data)

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

                # Calculate displacement from the last step
                displacement = p.data - state['prev_param']
                
                # Adapt the learning rate
                eta_adaptive = group['lr'] / torch.sqrt(1 + displacement**2 + group['eps'])
                
                # Store the current parameter value for the next step's displacement calculation
                state['prev_param'] = p.data.clone()

                # Apply the update
                p.data.add_(-eta_adaptive * grad)
        
        return loss

# ========================================
# 2. Training and Evaluation
# ========================================

def train_epoch(model, dataloader, optimizer, lr_scheduler, device, scaler):
    """
    Runs one full epoch of training.
    """
    model.train()
    total_loss = 0
    epoch_start_time = time.time()

    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Automatic Mixed Precision for faster training on modern GPUs
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss

        # Scale the loss and backpropagate
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
    """
    Evaluates the model on the validation set.
    """
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
    # Specify the optimizer and its learning rate
    parser = argparse.ArgumentParser(description="Fine-tune a transformer with LoRA.")
    parser.add_argument("--optimizer", type=str, required=True, choices=["AdamW", "IDAM"], help="Optimizer to use.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW or base LR (alpha) for IDAM.")
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "distilbert-base-uncased"
    dataset_name = "glue"
    subset_name = "sst2"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(os.environ.get("WORK", "."), "hf_cache")

    print(f"--- Starting Experiment: Optimizer={args.optimizer}, LR/Alpha={args.lr} ---")
    print(f"Device: {device}")

    # --- Data Loading and Preprocessing ---
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

    # --- Model Setup with LoRA ---
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"] # Target query and value layers in DistilBERT
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()
    
    # --- Optimizer and Scheduler Setup ---
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == "IDAM":
        optimizer = IDAM(model.parameters(), lr=args.lr)
    
    num_training_steps = args.epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Initialize the gradient scaler for mixed precision
    scaler = GradScaler()

    # --- Training Loop ---
    results = []
    total_training_time = 0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss, epoch_duration = train_epoch(model, train_dataloader, optimizer, lr_scheduler, device, scaler)
        val_accuracy = evaluate(model, eval_dataloader, device)
        
        total_training_time += epoch_duration
        
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Accuracy = {val_accuracy:.4f}, Duration = {epoch_duration:.2f}s")
        results.append({
            "epoch": epoch + 1, 
            "train_loss": train_loss, 
            "val_accuracy": val_accuracy,
            "epoch_duration_sec": epoch_duration
        })

    # --- Save Results ---
    results_df = pd.DataFrame(results)
    # Add a summary row for total time
    summary = pd.DataFrame([{"epoch": "total", "epoch_duration_sec": total_training_time}])
    results_df = pd.concat([results_df, summary], ignore_index=True)

    output_filename = os.path.join(output_dir, f"results_{args.optimizer}.csv")
    results_df.to_csv(output_filename, index=False)
    print(f"\n--- Experiment Complete ---")
    print(f"Saved final results to {output_filename}")

if __name__ == "__main__":
    main()
