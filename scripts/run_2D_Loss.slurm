#!/bin/bash -l
#SBATCH --account=hpca
#SBATCH --partition=lm_devel
#SBATCH --job-name=2D_Loss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --exclusive

echo "script started"
source venv_idam/bin/activate
python3 notebooks/2D_Loss.py
echo "script finished"

