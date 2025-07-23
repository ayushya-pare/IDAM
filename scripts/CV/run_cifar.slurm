#!/bin/bash
#SBATCH --job-name=cifar_hpo
#SBATCH --output=results/CV/cifar_hpo_%A_%a.out
#SBATCH --error=results/CV/cifar_hpo_%A_%a.err
#SBATCH --partition=sgpu_devel
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-7

# --- HPO Grid Search Setup ---
# We will test 8 combinations in total.
# 2 learning rates for each of the 4 optimizers.


# Define the hyperparameter search space
OPTIMIZERS=('SGD' 'IDAM' 'IDAM' 'IDAM' 'IDAM' 'IDAM' 'IDAM' 'IDAM')
VARIANTS=('' 'original' 'exp' 'tanh' 'log' 'sigmoid' 'cos' 'double_exp')
LEARNING_RATES=(0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1)


# Get the configuration for the current job
CURRENT_OPTIMIZER=${OPTIMIZERS[$SLURM_ARRAY_TASK_ID]}
CURRENT_VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
CURRENT_LR=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}

echo "--- Starting CIFAR-100 HPO Job ---"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Optimizer: $CURRENT_OPTIMIZER"
echo "Variant: $CURRENT_VARIANT"
echo "Learning Rate: $CURRENT_LR"
echo "--------------------------------"

# Activate your environment
source ../venv_idam/bin/activate

# Remove previous results
rm results/CV/*

# Construct the command
CMD="python scripts/CV/train_cifar.py \
    --optimizer $CURRENT_OPTIMIZER \
    --lr $CURRENT_LR \
    --wandb_project 'idam-cifar100-variants'"

# Add the variant flag only if it's an IDAM run
if [ "$CURRENT_OPTIMIZER" == "IDAM" ]; then
    CMD="$CMD --idam_variant $CURRENT_VARIANT"
fi

# Execute the command
eval $CMD

echo "--- Job Finished ---"
