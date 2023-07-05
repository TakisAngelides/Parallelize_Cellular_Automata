#!/bin/bash
#SBATCH -J njit_timed
#SBATCH -o njit_timed.txt
###SBATCH -e njit_timed.err
#SBATCH -p p100
#SBATCH --ntasks-per-node=32
#SBATCH -n 64
#SBATCH -N 2

srun python main.py
    