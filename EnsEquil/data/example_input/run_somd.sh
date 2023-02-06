#!/bin/bash
#SBATCH -o somd-array-gpu-%A.%a.out
#SBATCH -p main
#SBATCH -n 1
#SBATCH --time 24:00:00
#SBATCH --gres=gpu:1

lam=$1
echo "lambda is: " $lam

#sleep 15

srun /home/finlayclark/sire.app/bin/somd-freenrg -C ./input/sim.cfg -l $lam -p CUDA
