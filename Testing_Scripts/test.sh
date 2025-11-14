#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=test.txt
#SBATCH --partition=gpu
#SBATCH --gpus=a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=0:05:00	
#SBATCH --mail-type=ALL

module load miniconda
conda activate mjo_env_1

python test.py