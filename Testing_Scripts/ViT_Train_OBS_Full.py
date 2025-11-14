#!/usr/bin/env python
# coding: utf-8

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
from ViT_Unpatch import ViT

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
model_leadTms = "FullField2"
lead_time_width = 2

# --------------------
# Data Directories
# --------------------
mdl_directory = "/gpfs/gibbs/project/lu_lu/ax59/MJO_Project/Data/AllLeadTms/"
mdl_out_dir = "Result_01/"
obs_directory = "/gpfs/gibbs/project/lu_lu/ax59/MJO_Project/Data/AllLeadTms/"
obs_out_dir = "Result_02/"
bcor_directory = "Result_03/"
time_dir = "Data/Time/"

# --------------------
# Section 0: Class and Helper Function Definitions
# --------------------

class MDLDataset(torch.utils.data.Dataset):
    def __init__(self, mdl_dir, time_dir, lead_time):
        print("Loading MDL dataset into memory (NumPy arrays)...")
        self.width = lead_time
        # Load Features and Conditioning into NumPy arrays 
        features = []
        variable_names = ["TMQ", "FLUT", "U200", "U850", "TREFHT"]
        
        # Load data
        lead_time_vars = []
        for var_name in variable_names:
            file_path = os.path.join(mdl_dir, f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_{var_name}_leadTm1.nc")
            with Dataset(file_path) as f:
                lead_time_vars.append(f.variables[var_name][:])
            
        # Stack variables along a new channel dimension -> (samples, vars, lat, lon)
        features = np.stack(lead_time_vars, axis=1)

        # Concatenate all lead times into single large NumPy arrays
        self.features = features
        
        print(f"MDL dataset loaded. Total samples: {len(self.features)}")

    def __len__(self):
        return len(self.features) - self.width

    def __getitem__(self, idx):
        # Retrieve the data as NumPy arrays
        
        features = self.features[idx]
        targets = self.features[idx+self.width]
        
        # Convert only this specific sample to a tensor before returning
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32)
        )

class OBSDataset(torch.utils.data.Dataset):
    def __init__(self, obs_dir, time_dir, lead_time, indices=None):
        print("Loading OBS dataset into memory (NumPy arrays)...")
        self.width = lead_time

        # --- Load all data into full NumPy arrays first ---
        with Dataset(os.path.join(obs_dir, "CML2025_Step0C_OBS_remapped_90x180_daily_DJFM_nonFltr_PC1.nc")) as f:
            pc1 = f.variables["PC1"][:]
        with Dataset(os.path.join(obs_dir, "CML2025_Step0C_OBS_remapped_90x180_daily_DJFM_nonFltr_PC2.nc")) as f:
            pc2 = f.variables["PC2"][:]
        self.targets_np = np.stack([pc1, pc2], axis=-1)

        features = []
        variable_names = ["tcwv", "olr", "u200", "u850", "trefht"]

        lead_time_vars = []
        for var_name in variable_names:
            file_path = os.path.join(obs_dir, f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_{var_name}_leadTm1.nc")
            with Dataset(file_path) as f:
                lead_time_vars.append(f.variables[var_name][:])
        
        features = np.stack(lead_time_vars, axis=1)

        # --- Filter by indices if provided to create the final dataset ---
        if indices is not None:
            self.features = features[indices]
            print(f"OBS dataset subset loaded. Total samples: {len(self.features)-self.width}")
        else:
            self.features = features
            print(f"OBS dataset loaded. Total samples: {len(self.features)-self.width}")

    def __len__(self):
        return len(self.features) - self.width

    def __getitem__(self, idx):
        features = self.features[idx]
        targets = self.features[idx + self.width]
        
        # Convert only this specific sample to a tensor before returning
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32)
        )

class EarlyStopping:
    def __init__(self, threshold, patience):
        self.threshold = threshold
        self.patience = patience
        self.patience_count = 0
    def check_early_stop(self, test_loss_history):
        if len(test_loss_history) < self.patience:
            return False
        if min(test_loss_history[:-1]) - test_loss_history[-1] < self.threshold:
            self.patience_count += 1
            if self.patience_count >= self.patience:
                return True
        else:
            self.patience_count = 0
            return False

if __name__ == "__main__":
    
    # --------------------
    # Section 1: MDL Training
    # --------------------

    # Load dataset
    obs_dataset = OBSDataset(obs_dir=obs_directory, time_dir=time_dir, lead_time=lead_time_width)

    # Create DataLoaders
    batch_size = 121
    train_dataloader = torch.utils.data.DataLoader(obs_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)
    test_dataloader = train_dataloader

    # ViT Model
    model = ViT(
        image_size=(30, 180),
        patch_size=5,
        num_classes=2,
        dim=512,
        depth=10,
        heads=8,
        mlp_dim=1024,
        channels=5,
        dropout=0.0,
        emb_dropout=0.0
    )

    lr = 0.001
    epochs = 500
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.0001, factor=0.5, mode='min')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min = 1e-6)
    loss_fn = torch.nn.MSELoss()

    earlystopping = EarlyStopping(threshold=0.0001, patience=10)

    # Initial evaluation
    train_losses = []
    test_losses = []
    with torch.no_grad():
        model.eval()
        total_train_loss = 0.0
        for batch_data, batch_target in train_dataloader:
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            if batch_data.ndim == 3:
                batch_data = batch_data.unsqueeze(1)

            # Get the expected size from the model's configuration
            expected_h = model.image_height
            expected_w = model.image_width
            
            # Assert that the input data's dimensions match
            assert batch_data.shape[-2] == expected_h and batch_data.shape[-1] == expected_w, \
                f"Input data size mismatch! Got {batch_data.shape}, but model expects [B, C, {expected_h}, {expected_w}]"

            y = model(batch_data)
            loss = loss_fn(y, batch_target)
            total_train_loss += loss.item() * batch_data.size(0)
        avg_train_loss = total_train_loss / len(train_dataloader.dataset)
        total_test_loss = 0.0
        for batch_data, batch_target in test_dataloader:
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            pred_y = model(batch_data)
            loss = loss_fn(pred_y, batch_target)
            total_test_loss += loss.item() * batch_data.size(0)
        avg_test_loss = total_test_loss / len(test_dataloader.dataset)
        print(f"Initial Performance of Untrained Model: \nAvg Train Loss: {avg_train_loss}, Avg Test Loss: {avg_test_loss}")
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

    for i in range(epochs):
        total_train_loss = 0.0
        for batch_data, batch_target in train_dataloader:
            batch_data = batch_data.to(device)
            if batch_data.ndim == 3:
                batch_data = batch_data.unsqueeze(1)
            batch_target = batch_target.to(device)
            model.train()
            y = model(batch_data)
            loss = loss_fn(y, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_data.size(0)
        avg_train_loss = total_train_loss / len(train_dataloader.dataset)
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_target in test_dataloader:
                batch_data = batch_data.to(device)
                batch_target = batch_target.to(device)
                pred_y = model(batch_data)
                loss = loss_fn(pred_y, batch_target)
                total_test_loss += loss.item() * batch_data.size(0)
        avg_test_loss = total_test_loss / len(test_dataloader.dataset)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        #scheduler.step(avg_test_loss)
        scheduler.step()
        print(f"Epoch: {i + 1}, Avg Train Loss: {avg_train_loss}, Avg Test Loss: {avg_test_loss}, LR: {scheduler.get_last_lr()}")
        
        #if earlystopping.check_early_stop(test_losses):
        #    break



    mdl_model_path = mdl_out_dir + f"ViTTIMJO_FiLM_Andrew_MDL_leadTm{model_leadTms}_ensm{seed_num}.pth"
    torch.save(model, mdl_model_path)
    print(f"\nMDL Training Completed. Model saved to {mdl_model_path}")

    del mdl_dataset, train_dataloader, test_dataloader
    import gc; gc.collect()
    
    print("\nCombined MDL and OBS Transfer Learning Training Completed.")