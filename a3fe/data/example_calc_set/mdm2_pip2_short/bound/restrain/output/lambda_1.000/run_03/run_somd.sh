#!/bin/bash
#SBATCH -o somd-array-gpu-%A.%a.out
#SBATCH -p main
#SBATCH -n 1
#SBATCH --gres=gpu:1

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

lam=$1
echo "lambda is: " $lam

srun somd-freenrg -C somd.cfg -l $lam -p CUDA

