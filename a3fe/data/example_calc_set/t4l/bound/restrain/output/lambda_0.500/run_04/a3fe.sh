#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%A.%a.out

somd-freenrg -C somd.cfg -l 0.5 -p CUDA
