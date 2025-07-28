#!/bin/bash
#SBATCH --job-name=cifar100_training
#SBATCH --output=results/CV/cifar_hpo_%A_%a.out
#SBATCH --error=results/CV/cifar_hpo_%A_%a.err

#SBATCH --partition=sgpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4             # Number of CPU threads for data loading
#SBATCH --mem=32G                     # Memory needed
#SBATCH --time=00:30:00               
#SBATCH --array=0-2

## Optimizers and parameters
OPTIMIZERS=('IDAM' 'Adam' 'SGD')
LEARNING_RATES=(0.1 0.001 0.1)

## Get the configuration for the current job
CURRENT_OPTIMIZER=${OPTIMIZERS[$SLURM_ARRAY_TASK_ID]}
CURRENT_LR=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}
echo "--- Starting CIFAR-100 HPO Job ---"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Optimizer: $CURRENT_OPTIMIZER"
echo "Learning Rate: $CURRENT_LR"
echo "--------------------------------"

## Activate your environment
module load cuda/12.6
conda deactivate
source ../venv_idam/bin/activate

## Construct the command
python scripts/CV/train_cifar.py --optimizer $CURRENT_OPTIMIZER --lr $CURRENT_LR --wandb_project 'cifar100_training'
echo "--- Job Finished ---"
