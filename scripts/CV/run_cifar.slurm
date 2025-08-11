#!/bin/bash
#SBATCH --job-name=cifar100_training
#SBATCH --output=results/CV/cifar_hpo_%A_%a.out
#SBATCH --error=results/CV/cifar_hpo_%A_%a.err

#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8             # Number of CPU threads for data loading
##SBATCH --mem=32G                     # Memory needed
#SBATCH --exclusive
#SBATCH --time=00:20:00               
#SBATCH --array=0-1

## Optimizers and parameters
OPTIMIZERS=('IDAM' 'custom_Adam')
LEARNING_RATES=(0.1 0.001)

## Get the configuration for the current job
CURRENT_OPTIMIZER=${OPTIMIZERS[$SLURM_ARRAY_TASK_ID]}
CURRENT_LR=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}
echo "--- Starting CIFAR-100 HPO Job ---"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Optimizer: $CURRENT_OPTIMIZER"
echo "Learning Rate: $CURRENT_LR"
echo "--------------------------------"

## Activate your environment
##module load cuda/12.6
conda deactivate
source ../venv_idam/bin/activate

## Construct the command
python scripts/CV/train_cifar.py --optimizer $CURRENT_OPTIMIZER --lr $CURRENT_LR --epochs 20 --wandb_project 'cifar100_training_doover'
echo "--- Job Finished ---"
