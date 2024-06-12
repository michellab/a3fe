#!/bin/bash
#SBATCH -o somd-array-gpu-%A.%a.out
#SBATCH -p GTX980,RTX3080
#SBATCH --exclude node01,node02,node04,node06
#SBATCH -n 1
#SBATCH --gres=gpu:1

module load cuda
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

lam=$1
echo "lambda is: " $lam

srun somd-freenrg -C somd.cfg -l $lam -p CUDA

