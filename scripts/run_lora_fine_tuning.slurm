#!/bin/bash -l
#SBATCH --job-name=idam_llm_test
#SBATCH --account=hpca

#SBATCH --partition=sgpu_short
#SBATCH --gres=gpu:1
#SBATCH --threads-per-core=1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8 
#SBATCH --array=0-1
#SBATCH --exclusive

#SBATCH --output=results/job_output_%A_%a.out
#SBATCH --error=results/job_output_%A_%a.err

# --- Setup ---
# This job array will run two tasks, one for each optimizer.
# Task 0: AdamW
# Task 1: IDAM

echo "======================================================"
echo "Job ID: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "======================================================"

# Activate your Python virtual environment
source venv/bin/activate

# --- Run the appropriate training script based on the task ID ---
case $SLURM_ARRAY_TASK_ID in
  0)
    # This is the first job (ID=0), run with AdamW
    echo "Running experiment with AdamW optimizer"
    python scripts/train_comparison.py \
      --optimizer AdamW \
      --lr 1e-4 \
      --batch_size 32 \
      --epochs 3
    ;;
  1)
    # This is the second job (ID=1), run with IDAM
    echo "Running experiment with IDAM optimizer"
    python scripts/train_comparison.py \
      --optimizer IDAM \
      --lr 0.05 \
      --batch_size 32 \
      --epochs 3
    ;;
esac

echo "======================================================"
echo "Job finished"
echo "======================================================"
