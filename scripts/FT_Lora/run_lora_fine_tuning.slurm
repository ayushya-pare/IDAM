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
# Task 0: AdamW
# Task 1: IDAM


# Activate virtual environment
source ../venv_idam/bin/activate

echo "======================================================"
echo "Job started"
echo "======================================================"


# --- Run the appropriate training script based on the task ID ---
case $SLURM_ARRAY_TASK_ID in
  0)
    echo "Running experiment with AdamW optimizer"
    python scripts/train_lora.py \
      --optimizer AdamW \
      --lr 1e-4 \
      --batch_size 32 \
      --epochs 5
    ;;
  1)
    echo "Running experiment with IDAM optimizer"
    python scripts/train_lora.py \
      --optimizer IDAM \
      --lr 0.1 \
      --batch_size 32 \
      --epochs 5
    ;;
esac

echo "======================================================"
echo "Job finished"
echo "======================================================"

