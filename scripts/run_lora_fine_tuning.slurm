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
# Define the optimizers to test
OPTIMIZERS=(AdamW IDAM)

# Select the optimizer for the current array task
CURRENT_OPTIMIZER=${OPTIMIZERS[$SLURM_ARRAY_TASK_ID]}

echo "======================================================"
echo "Starting SLURM job"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Optimizer: $CURRENT_OPTIMIZER"
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "======================================================"

# --- Run the training script ---
source venv_idam/bin/activate
python scripts/train_lora.py --optimizer $CURRENT_OPTIMIZER

source venv_idam/bin/activate
python3 notebooks/2D_Loss.py
echo "script finished"

echo "======================================================"
echo "Job finished"
echo "======================================================"