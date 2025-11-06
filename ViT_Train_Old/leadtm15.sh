#!/bin/bash

#SBATCH --job-name=MJO_LeadTm15
#SBATCH --output=leadtm15test.txt
#SBATCH --partition=gpu
#SBATCH --gpus=a5000:1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=5:00:00	
#SBATCH --mail-type=ALL

module load miniconda
conda activate mjo_env_1

python ViT_Train_Time15.py