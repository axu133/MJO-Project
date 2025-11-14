#!/usr/bin/env python
# coding: utf-8

# ## ViT for MJO Index Prediction - Combined MDL Train and OBS Transfer Learning
# #### Andrew Xu
# #### Created June 22, 2025

import matplotlib.pyplot as plt
import numpy as np
import torch
from netCDF4 import Dataset
#from vit_pytorch import ViT
import os
import random
import copy
import csv
import pandas as pd
from ViT_FiLM import ViT

import xarray as xr

# --------------------
# General Setup
# --------------------

deBug = True
seed_num = 1
np.random.seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
torch.cuda.is_available()
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
model_leadTms = "10-30More_schedulerPlateau"
lead_times_to_load = [10,13,15,18,20,23,25,28,30]

# --------------------
# Data Directories
# --------------------
mdl_directory = "/gpfs/gibbs/project/lu_lu/ax59/MJO_Project/Data/AllLeadTms/"
mdl_out_dir = "Result_01/"
obs_directory = "/gpfs/gibbs/project/lu_lu/ax59/MJO_Project/Data/AllLeadTms/"
obs_out_dir = "Result_02/"
bcor_directory = "Result_03/"
time_dir = "Data/Time/"

file_path = "/gpfs/gibbs/project/lu_lu/ax59/MJO_Project/Data/AllLeadTms/CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_tcwv_leadTm10.nc"

ds = xr.open_dataset(file_path, use_cftime=True)

print(ds)

df = ds.to_dataframe().reset_index()

print(ds)
print("Data variables:", list(ds.data_vars))
print("Coordinates:", list(ds.coords))