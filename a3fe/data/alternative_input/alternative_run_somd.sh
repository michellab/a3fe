#!/bin/bash

#SBATCH --account=qt
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --exclude=node018
#SBATCH --mem=50G
#SBATCH --output=somd-array-gpu-%A.%a.out

lam=$1
echo "lambda is: " $lam

srun somd-freenrg -C somd.cfg -l $lam -p CUDA
