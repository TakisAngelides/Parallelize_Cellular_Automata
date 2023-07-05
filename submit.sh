#!/bin/bash

#SBATCH --job-name=timing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=p100
#SBATCH --error=job.%j.err
#SBATCH --output=job.%j.out

python main.py
