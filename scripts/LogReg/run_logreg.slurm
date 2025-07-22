#!/bin/bash
#SBATCH --job-name=idam_variants
#SBATCH --output=results/LogReg/variants_%A_%a.out
#SBATCH --error=results/LogReg/variants_%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --array=0-6

# --- Experiment Grid ---
# Task 0: Adam (Baseline)
# Task 1: SGD (Baseline)
# Task 2: IDAM (Original)
# Task 3: IDAM (Linear Decay)
# Task 4: IDAM (Exponential Decay)
# Task 5: IDAM (Logarithmic Damping)
# Task 6: IDAM-V (Second-Order Variance)

# Define the configurations for each task
OPTIMIZERS=('Adam' 'SGD' 'IDAM' 'IDAM')
VARIANTS=('' '' 'original' 'exp')
LEARNING_RATES=(0.0005 0.0005 0.0005 0.0005)

# Get the configuration for the current job
CURRENT_OPTIMIZER=${OPTIMIZERS[$SLURM_ARRAY_TASK_ID]}
CURRENT_VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
CURRENT_LR=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}

echo "--- Starting Variant Test Job ---"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Optimizer: $CURRENT_OPTIMIZER"
echo "Variant: $CURRENT_VARIANT"
echo "Learning Rate: $CURRENT_LR"
echo "-------------------------------"

# Activate your environment
source venv_idam/bin/activate

# Construct the command
CMD="python scripts/LogReg/train_logreg.py \
    --optimizer $CURRENT_OPTIMIZER \
    --lr $CURRENT_LR \
    --wandb_project 'idam-variants-benchmark'"

# Add the variant flag only if it's an IDAM run
if [ "$CURRENT_OPTIMIZER" == "IDAM" ]; then
    CMD="$CMD --idam_variant $CURRENT_VARIANT"
fi

# Execute the command
eval $CMD

echo "--- Job Finished ---"
