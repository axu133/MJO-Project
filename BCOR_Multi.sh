#!/bin/bash

#SBATCH --job-name=BCORMulti
#SBATCH --output=BCORMulti.txt
#SBATCH --partition=gpu
#SBATCH --gpus=a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00	
#SBATCH --mail-type=ALL

module load miniconda
conda activate mjo_env_1

python BCOR_Multi.py