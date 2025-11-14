#!/bin/bash

#SBATCH --job-name=MJO_LeadTm15_FiLM
#SBATCH --output=leadtm15FiLM_output.txt
#SBATCH --partition=gpu
#SBATCH --gpus=a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00	
#SBATCH --mail-type=ALL

module load miniconda
conda activate mjo_env_1

python ViT_Train_Film_15.py