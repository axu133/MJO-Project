#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import torch
from netCDF4 import Dataset, num2date
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
model_leadTms = "FullField2NoCV"
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
        print("Loading MDL dataset and processing independent runs...")
        self.width = lead_time
        
        # Load Data
        variable_names = ["TMQ", "FLUT", "U200", "U850", "TREFHT"]
        time_objects = None
        lead_time_vars = []
        
        for i, var_name in enumerate(variable_names):
            file_path = os.path.join(mdl_dir, f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_{var_name}_leadTm1.nc")
            with Dataset(file_path) as f:
                lead_time_vars.append(f.variables[var_name][:])
                if i == 0:
                    time_var = f.variables['time']
                    time_objects = num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
                    
        all_features_raw = np.stack(lead_time_vars, axis=1)
        self.times = time_objects 
        print(f"MDL raw data loaded. {len(all_features_raw)} total entries.")

        # Boundaries of each run
        run_boundaries = np.where(self.times[1:] < self.times[:-1])[0]
        
        # Create a list of start/end indices for each run
        run_chunks_indices = []
        start_idx = 0
        for boundary_idx in run_boundaries:
            end_idx = boundary_idx + 1 # The chunk includes this index
            run_chunks_indices.append((start_idx, end_idx))
            start_idx = end_idx # The next chunk starts after the boundary
        run_chunks_indices.append((start_idx, len(all_features_raw))) # Add the last run
        
        print(f"Found {len(run_chunks_indices)} independent runs in the file.")

        # Process runs individually
        self.valid_pairs = []
        for run_start, run_end in run_chunks_indices:
            run_features = all_features_raw[run_start:run_end]
            run_times = self.times[run_start:run_end]

            # Find gaps
            chunk_start_idx_seasonal = 0
            if len(run_times) < 2: continue # Skip if the run is too short

            seasonal_gaps = np.where((run_times[1:] - run_times[:-1]).astype('timedelta64[D]').astype(int) > 1)[0]
            
            chunk_end_indices_seasonal = list(seasonal_gaps)
            chunk_end_indices_seasonal.append(len(run_times) - 1)

            for gap_idx_seasonal in chunk_end_indices_seasonal:
                chunk_end_slice = gap_idx_seasonal + 1
                
                season_features = run_features[chunk_start_idx_seasonal:chunk_end_slice]
                season_times = run_times[chunk_start_idx_seasonal:chunk_end_slice]
                
                # Create input-target pairs
                num_in_season = len(season_features)
                for i in range(num_in_season - self.width): # feature_time, target_time for debugging
                    feature = season_features[i]
                    target = season_features[i + self.width]
                    feature_time = season_times[i]
                    target_time = season_times[i + self.width]
                    self.valid_pairs.append((feature, target, feature_time, target_time))
                
                chunk_start_idx_seasonal = gap_idx_seasonal + 1

        print(f"MDL dataset fully processed. Found {len(self.valid_pairs)} valid input-target pairs across all runs.")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        feature, target, feature_time, target_time = self.valid_pairs[idx]
        
        return (
            torch.tensor(feature, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )

class OBSDataset(torch.utils.data.Dataset):
    def __init__(self, obs_dir, time_dir, lead_time, indices=None):
        print("Loading OBS dataset and processing seasonal chunks...")
        self.width = lead_time

        # Load Data
        variable_names = ["tcwv", "olr", "u200", "u850", "trefht"]
        time_objects = None
        lead_time_vars = []
        
        for i, var_name in enumerate(variable_names):
            file_path = os.path.join(obs_dir, f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_{var_name}_leadTm1.nc")
            with Dataset(file_path) as f:
                lead_time_vars.append(f.variables[var_name][:])
                if i == 0:
                    time_var = f.variables['time']
                    time_objects = num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
        
        all_features_full = np.stack(lead_time_vars, axis=1)
        all_times_full = time_objects

        # Subset indices
        if indices is not None:
            all_features = all_features_full[indices]
            self.times = all_times_full[indices] 
            print(f"OBS dataset subset applied. Processing {len(all_features)} time steps for this fold.")
        else:
            all_features = all_features_full
            self.times = all_times_full
            print(f"OBS dataset loaded. Processing {len(all_features)} time steps.")

        # Find gaps
        self.valid_pairs = []
        chunk_start_idx = 0
        
        if len(self.times) == 0:
            print("Warning: No data to process (empty indices). Dataset will be empty.")
            return

        gaps = np.where((self.times[1:] - self.times[:-1]).astype('timedelta64[D]').astype(int) > 1)[0]

        for gap_idx in np.append(gaps, len(self.times) - 1):
            chunk_end_idx = gap_idx + 1
            
            chunk_features = all_features[chunk_start_idx:chunk_end_idx]
            chunk_times = self.times[chunk_start_idx:chunk_end_idx]
            
            # Create input-target pairs
            num_in_chunk = len(chunk_features)
            for i in range(num_in_chunk - self.width): # feature_time, target_time for debugging
                feature = chunk_features[i]
                target = chunk_features[i + self.width]
                feature_time = chunk_times[i]
                target_time = chunk_times[i + self.width]
                self.valid_pairs.append((feature, target, feature_time, target_time))
            
            chunk_start_idx = chunk_end_idx
            
        print(f"OBS dataset processed. Found {len(self.valid_pairs)} valid input-target pairs.")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        feature, target, feature_time, target_time = self.valid_pairs[idx]
        
        return (
            torch.tensor(feature, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
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
    """
    # --------------------
    # Section 1: MDL Training
    # --------------------

    # Load dataset
    mdl_dataset = MDLDataset(mdl_dir=mdl_directory, time_dir=time_dir, lead_time=lead_time_width)

    # Split into training and testing sets
    mdl_total_samples = len(mdl_dataset)
    mdl_train_size = int(0.8 * mdl_total_samples) - int(0.8 * mdl_total_samples) % 121
    mdl_test_size = mdl_total_samples - mdl_train_size
    mdl_train_dataset, mdl_test_dataset = torch.utils.data.random_split(mdl_dataset, [mdl_train_size, mdl_test_size])

    # Create DataLoaders
    batch_size = 121
    train_dataloader = torch.utils.data.DataLoader(mdl_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(mdl_test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=4)

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
    """
    # --------------------
    # Section 2: OBS Transfer Learning (as in ViT Train 2.py)
    # --------------------

    # General setup

    mdl_model_path = mdl_out_dir + f"ViTTIMJO_FiLM_Andrew_MDL_leadTmFullField2_ensm{seed_num}.pth"

    lat = 30
    lon = 180
    var = 5
    batch_size = 121

    obs_dataset = OBSDataset(obs_dir=obs_directory, time_dir=time_dir, lead_time=lead_time_width)

    total_samples = len(obs_dataset)
    print(f"Total OBS samples across all lead times: {total_samples}")

    obs_train_dataset, obs_test_dataset = torch.utils.data.random_split(obs_dataset, [int(0.8 * total_samples), total_samples - int(0.8 * total_samples)])
        
    # Transfer Learning Training
    lr = 1e-4
    epochs = 300
    lr_warmup_length = 5
    compiled_train_losses = []
    compiled_val_losses = []
    #for index, round in enumerate(train_val_rounds):
        
    train_losses = []
    val_losses = []
    train_dataloader = torch.utils.data.DataLoader(obs_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(obs_test_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)

    model = torch.load(mdl_model_path, weights_only=False).to(device) 
    batches_per_dataset = len(train_dataloader)
    print(f"batches per dataset: {batches_per_dataset}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.0001, factor=0.5, mode='min')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min = 1e-6)
    
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0 = 50,
    T_mult = 2,
    eta_min = 1e-7
    )
    """
    
    loss_fn = torch.nn.MSELoss()
    earlystopping = EarlyStopping(threshold=0.0001, patience=10)
    best_validation_loss = float("inf")
    best_model = None
    model.eval()
    
    with torch.no_grad():
        total_train_loss = 0.0
        for batch_data, batch_target in train_dataloader:
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            if batch_data.ndim == 3:
                batch_data = batch_data.unsqueeze(1)
            y = model(batch_data)
            loss = loss_fn(y, batch_target)
            total_train_loss += loss.item() * batch_data.size(0)
        avg_train_loss = total_train_loss / len(train_dataloader.dataset)
        total_val_loss = 0.0
        for batch_data, batch_target in val_dataloader:
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            if batch_data.ndim == 3:
                batch_data = batch_data.unsqueeze(1)
            pred_y = model(batch_data)
            loss = loss_fn(pred_y, batch_target)
            total_val_loss += loss.item() * batch_data.size(0)
        avg_val_loss = total_val_loss / len(val_dataloader.dataset)
        print(f"Performance Before Training: Avg Train Loss: {avg_train_loss}, Avg Val Loss: {avg_val_loss}")
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
    for i in range(epochs):
        total_train_loss = 0.0
        if i < lr_warmup_length:
            warmup_lr = lr * (i + 1) / (lr_warmup_length + 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=warmup_lr, weight_decay=1e-5)
        if i == lr_warmup_length:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.0001, factor=0.5, mode='min')
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min = 1e-6)
            """
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = 50,
            T_mult = 2,
            eta_min = 1e-7
            )
            """
            
        for batch_data, batch_target in train_dataloader:
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            if batch_data.ndim == 3:
                batch_data = batch_data.unsqueeze(1)
            model.train()
            y = model(batch_data)
            loss = loss_fn(y, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_data.size(0)
        avg_train_loss = total_train_loss / len(train_dataloader.dataset)
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_target in val_dataloader:
                batch_data = batch_data.to(device)
                batch_target = batch_target.to(device)    
                if batch_data.ndim == 3:
                    batch_data = batch_data.unsqueeze(1)    
                pred_y = model(batch_data)
                loss = loss_fn(pred_y, batch_target)
                total_val_loss += loss.item() * batch_data.size(0)
        avg_val_loss = total_val_loss / len(val_dataloader.dataset)
        if i >= lr_warmup_length:
            #scheduler.step(avg_val_loss)
            scheduler.step()
        print(f"Epoch: {i + 1}, Avg Train Loss: {avg_train_loss}, Avg Val Loss: {avg_val_loss}, LR: {scheduler.get_last_lr()}")
        if avg_val_loss < best_validation_loss:
            best_validation_loss = avg_val_loss
            best_model = copy.deepcopy(model)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        #if earlystopping.check_early_stop(val_losses):
        #    break
            
    model = best_model
    model.eval()
    torch.save(model, obs_out_dir + f"ViTTIMJO_FiLM_Andrew_OBS_leadTm{model_leadTms}_ensm{seed_num}_round1.pth")
    compiled_train_losses.append(train_losses)
    compiled_val_losses.append(val_losses)

    print("\nCombined MDL and OBS Transfer Learning Training Completed.")