#!/bin/bash
#SBATCH --job-name=cifar_hpo
##SBATCH --output=results/CV/cifar_hpo_%A_%a.out
#SBATCH --error=results/CV/cifar_hpo_%A_%a.err
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
##SBATCH --array=0-1

# Define the hyperparameter search space
OPTIMIZERS=('IDAM')
LEARNING_RATES=(0.1)


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
#rm results/CV/*

# Construct the command
CMD="python scripts/CV/train_cifar.py \
    --optimizer $CURRENT_OPTIMIZER \
    --lr $CURRENT_LR \
    --wandb_project 'cifar-100-benchmark'"


# Execute the command
eval $CMD

echo "--- Job Finished ---"
