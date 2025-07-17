# File: scripts/train_lora.py

import os
import argparse
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# ========================================
# 1. IDAM Optimizer
# ========================================
class IDAM(torch.optim.Optimizer):
    def __init__(self, params, alpha=0.1, eps=1e-8):
        defaults = dict(alpha=alpha, eps=eps)
        super(IDAM, self).__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
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

                # Adaptive learning rate based on displacement
                displacement = p.data - state['prev_param']
                eta_adaptive = group['alpha'] / torch.sqrt(1 + displacement**2 + group['eps'])
                
                # Update previous parameter state
                state['prev_param'] = p.data.clone()

                # Update parameter
                p.data.add_(-eta_adaptive * grad)
        
        return loss

# ========================================
# 2. Training and Evaluation
# ========================================

def train_epoch(model, dataloader, optimizer, lr_scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

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
    parser = argparse.ArgumentParser(description="Fine-tune a transformer with LoRA using different optimizers.")
    parser.add_argument("--optimizer", type=str, required=True, choices=["AdamW", "IDAM"], help="Optimizer to use.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha (base learning rate) for IDAM.")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "distilbert-base-uncased"
    dataset_name = "glue"
    subset_name = "sst2"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(os.environ.get("WORK", "."), "hf_cache")

    print(f"Using device: {device}")
    print(f"Running experiment with optimizer: {args.optimizer}")

    # Load and preprocess data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw_datasets = load_dataset(dataset_name, subset_name, cache_dir=cache_dir)

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=args.batch_size)

    # Setup Model with LoRA
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()
    
    # Setup Optimizer and Scheduler
    if args.optimizer == "AdamW":
        optimizer = AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == "IDAM":
        optimizer = IDAM(model.parameters(), alpha=args.alpha)
    
    num_training_steps = args.epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Training Loop
    results = []
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss = train_epoch(model, train_dataloader, optimizer, lr_scheduler, device)
        val_accuracy = evaluate(model, eval_dataloader, device)
        
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}")
        results.append({"epoch": epoch + 1, "train_loss": train_loss, "val_accuracy": val_accuracy})

    # Save results
    results_df = pd.DataFrame(results)
    output_filename = os.path.join(output_dir, f"results_{args.optimizer}.csv")
    results_df.to_csv(output_filename, index=False)
    print(f"\nSaved final results to {output_filename}")
    
    # Save the trained adapter
    adapter_path = os.path.join(output_dir, f"adapter_{args.optimizer}")
    model.save_pretrained(adapter_path)
    print(f"Saved LoRA adapter to {adapter_path}")

if __name__ == "__main__":
    main()