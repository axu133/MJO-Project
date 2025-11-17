#!/bin/bash

#SBATCH --job-name=FNO_MJO
#SBATCH --output=FNOoutput.txt
#SBATCH --partition=gpu
#SBATCH --gpus=a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=48:00:00	
#SBATCH --mail-type=ALL

module load miniconda
conda activate mjo_env_1

python FNO_Train_Full.py