#!/bin/bash
#SBATCH -J main_timed
#SBATCH -o main_timed.txt
###SBATCH -e main_timed.err
#SBATCH -p p100
#SBATCH --ntasks-per-node=32
#SBATCH -n 64
#SBATCH -N 2

srun python main.py
    