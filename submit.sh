#!/bin/bash
#SBATCH -J numba_gpu_timed
#SBATCH -o numba_gpu_timed.txt
#SBATCH -e numba_gpu_timed.err
#SBATCH -p p100
#SBATCH --cpus-per-gpu=32
#SBATCH --gres=gpu:1
#SBATCH -n 64
#SBATCH -N 1

srun python main.py
    