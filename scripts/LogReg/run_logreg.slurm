#!/bin/bash
#SBATCH --job-name=logreg_benchmark
#SBATCH --output=results/LogReg/logreg_output_%A_%a.out
#SBATCH --error=results/LogReg/logreg_output_%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --array=0-2

# --- Setup ---
# Task 0: Adam
# Task 1: SGD
# Task 2: IDAM

OPTIMIZERS=(Adam SGD IDAM)
CURRENT_OPTIMIZER=${OPTIMIZERS[$SLURM_ARRAY_TASK_ID]}

# Use different learning rates for each optimizer for a fair comparison
case $SLURM_ARRAY_TASK_ID in
  0) LR=0.001 ;; # Adam
  1) LR=0.1   ;; # SGD
  2) LR=0.15  ;; # IDAM
esac

echo "--- Starting Job: Optimizer=$CURRENT_OPTIMIZER, LR=$LR ---"

# Activate your environment
source venv_idam/bin/activate

# Run the training script
python scripts/train_logreg.py \
    --optimizer $CURRENT_OPTIMIZER \
    --lr $LR \
    --wandb_project "idam-logreg-benchmark"

echo "--- Job Finished ---"

