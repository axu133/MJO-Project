#!/bin/bash

#SBATCH --job-name=FiLM_MJO_LeadTm10-15
#SBATCH --output=film_leadtmFull.txt
#SBATCH --partition=gpu
#SBATCH --gpus=a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00	
#SBATCH --mail-type=ALL

module load miniconda
conda activate mjo_env_1

python Vit_Train_Film_Full.py