#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_freenrg-MBAR-run_02_100_end_0_start.out

analyse_freenrg mbar -i /home/roy/software/deve/a3fe/runtest/t4l/bound/restrain/output/lambda*/run_02/simfile_truncated_100_end_0_start.dat -p 100 --overlap --output /home/roy/software/deve/a3fe/runtest/t4l/bound/restrain/output/freenrg-MBAR-run_02_100_end_0_start.dat
