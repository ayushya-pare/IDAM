#!/bin/bash
#SBATCH --job-name=benchmark_optimizers
#SBATCH --output=results/benchmarks/bench_optim_%A_%a.out
#SBATCH --error=results/benchmarks/bench_optim_%A_%a.err

#SBATCH --partition=lm_devel
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8             # Number of CPU threads for data loading
##SBATCH --mem=32G                     # Memory needed
#SBATCH --exclusive
#SBATCH --time=00:10:00               
##SBATCH --array=0-1

conda deactivate
source ../venv_idam/bin/activate

python scripts/benchmarks/bench_optim.py

