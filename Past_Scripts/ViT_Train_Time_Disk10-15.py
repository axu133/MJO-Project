#!/usr/bin/env python
# coding: utf-8

# ## ViT for MJO Index Prediction - Combined MDL Train and OBS Transfer Learning
# #### Andrew Xu
# #### Created June 22, 2025

import matplotlib.pyplot as plt
import numpy as np
import torch
from netCDF4 import Dataset
from vit_pytorch import ViT
import os
import random
import copy
import csv
import pandas as pd

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
leadTm = "11-15"

# --------------------
# Data Directories
# --------------------
mdl_directory = "/gpfs/gibbs/project/lu_lu/bec32/ML_for_MJO/Data/AllLeadTms/"
mdl_out_dir = "Result_01/"
obs_directory = "/gpfs/gibbs/project/lu_lu/bec32/ML_for_MJO/Data/AllLeadTms/"
obs_out_dir = "Result_02/"
bcor_directory = "Result_03/"
time_dir = "Data/Time/"

# --------------------
# Section 1: MDL Training (as in ViT Train.py)
# --------------------

class MDLDiskDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that loads data directly from NetCDF files on disk
    to avoid storing the entire dataset in RAM.
    """
    def __init__(self, mdl_dir, time_dir, lead_times):
        self.mdl_dir = mdl_dir
        self.time_dir = time_dir
        self.lead_times = lead_times
        
        # Load the target data (PC1, PC2), which is small enough to keep in memory.
        with Dataset(os.path.join(self.mdl_dir, "CML2025_Step0C_MDL_remapped_90x180_daily_DJFM_nonFltr_PC1.nc")) as f:
            pc1 = f.variables["PC1"][:]
        with Dataset(os.path.join(self.mdl_dir, "CML2025_Step0C_MDL_remapped_90x180_daily_DJFM_nonFltr_PC2.nc")) as f:
            pc2 = f.variables["PC2"][:]
        self.targets = np.stack([pc1, pc2], axis=-1)

        # Scan files to create a map from a global index to a specific file and index within that file.
        self.index_map = []
        self.file_paths = {}
        total_samples = 0
        
        print("Scanning files to create index map")
        for lead_tm in self.lead_times:
            # Store file paths for this lead time
            self.file_paths[lead_tm] = {
                "TMQ": os.path.join(self.mdl_dir, f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_TMQ_leadTm{lead_tm}.nc"),
                "FLUT": os.path.join(self.mdl_dir, f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_FLUT_leadTm{lead_tm}.nc"),
                "U200": os.path.join(self.mdl_dir, f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_U200_leadTm{lead_tm}.nc"),
                "U850": os.path.join(self.mdl_dir, f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_U850_leadTm{lead_tm}.nc"),
                "TREFHT": os.path.join(self.mdl_dir, f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_TREFHT_leadTm{lead_tm}.nc"),
                "LEADTM": os.path.join(self.time_dir, f"MDLleadTm{lead_tm}.nc")
            }
            # Get the number of samples in the first variable's file to determine the length
            with Dataset(self.file_paths[lead_tm]["TMQ"]) as f:
                num_in_file = f.variables["TMQ"].shape[0]
                for i in range(num_in_file):
                    # The global index for the target corresponds to the total samples seen so far + local index
                    target_idx = i
                    self.index_map.append((lead_tm, i, target_idx))
            total_samples += num_in_file
        print(f"Map created. Total samples across all lead times: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # 1. Use the map to find which lead time and local index to load.
        lead_tm, local_idx, target_idx = self.index_map[idx]
        
        # 2. Open the corresponding files and load only the required slice of data.
        paths = self.file_paths[lead_tm]
        with Dataset(paths["TMQ"]) as f: tmq = f.variables["TMQ"][local_idx, :, :]
        with Dataset(paths["FLUT"]) as f: flut = f.variables["FLUT"][local_idx, :, :]
        with Dataset(paths["U200"]) as f: u200 = f.variables["U200"][local_idx, :, :]
        with Dataset(paths["U850"]) as f: u850 = f.variables["U850"][local_idx, :, :]
        with Dataset(paths["TREFHT"]) as f: trefht = f.variables["TREFHT"][local_idx, :, :]
        with Dataset(paths["LEADTM"]) as f: leadtm = f.variables["LEADTM"][local_idx, :, :]
            
        # 3. Stack the variables for this single sample.
        # Original shape: (lat, lon), stacked to (lat, lon, var)
        data_sample = np.stack([tmq, flut, u200, u850, trefht, leadtm], axis=-1)
        
        # 4. Transpose to the format your model expects: (var, lat, lon)
        data_sample = data_sample.transpose(2, 0, 1)

        # 5. Get the corresponding target.
        target_sample = self.targets[target_idx]
        
        # 6. Convert to tensors for PyTorch.
        x = torch.tensor(data_sample, dtype=torch.float32)
        y = torch.tensor(target_sample, dtype=torch.float32)
        
        return x, y

# Define the lead times you want to process
# This can be expanded to range(1, 41) for your full dataset
lead_times_to_load = range(11, 16) 

# Create the memory-efficient dataset
mdl_dataset = MDLDiskDataset(mdl_dir=mdl_directory, time_dir=time_dir, lead_times=lead_times_to_load)

# --- The rest of your script remains largely the same ---

batch_size = 121*5

# Split into training and testing sets
mdl_total_samples = len(mdl_dataset)
mdl_train_size = int(0.8 * mdl_total_samples) - int(0.8 * mdl_total_samples) % batch_size
mdl_test_size = mdl_total_samples - mdl_train_size
mdl_train_dataset, mdl_test_dataset = torch.utils.data.random_split(mdl_dataset, [mdl_train_size, mdl_test_size])

# Create DataLoaders
# Set num_workers > 0 to load data in parallel, which is highly recommended with this on-disk approach
train_dataloader = torch.utils.data.DataLoader(mdl_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(mdl_test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

# ViT Model
model = ViT(
    image_size=180,
    patch_size=15,
    num_classes=2,
    dim=256,
    depth=6,
    heads=8,
    mlp_dim=512,
    channels=6,
    dropout=0.1,
    emb_dropout=0.1
)

lr = 0.0005
epochs = 500
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.0001, factor=0.5, mode='min')
loss_fn = torch.nn.MSELoss()

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
    scheduler.step(avg_test_loss)
    print(f"Epoch: {i + 1}, Avg Train Loss: {avg_train_loss}, Avg Test Loss: {avg_test_loss}, LR: {scheduler.get_last_lr()}")
    if earlystopping.check_early_stop(test_losses):
        break

mdl_model_path = mdl_out_dir + f"ViTTIMJO_Andrew_MDL_leadTm{leadTm}_ensm{seed_num}.pth"
torch.save(model, mdl_model_path)
print(f"\nMDL Training Completed. Model saved to {mdl_model_path}")

del mdl_dataset, train_dataloader, test_dataloader
import gc; gc.collect()

# --------------------
# Section 2: OBS Transfer Learning (as in ViT Train 2.py)
# --------------------

class OBSDiskDataset(torch.utils.data.Dataset):
    """
    A memory-efficient Dataset for observational (OBS) data that reads
    samples directly from disk and can represent a specific subset of
    indices for cross-validation.
    """
    def __init__(self, obs_dir, time_dir, lead_times, indices=None):
        self.obs_dir = obs_dir
        self.time_dir = time_dir
        self.lead_times = lead_times
        
        # Load the target data (PC1, PC2) into memory
        with Dataset(os.path.join(self.obs_dir, "CML2025_Step0C_OBS_remapped_90x180_daily_DJFM_nonFltr_PC1.nc")) as f:
            pc1 = f.variables["PC1"][:]
        with Dataset(os.path.join(self.obs_dir, "CML2025_Step0C_OBS_remapped_90x180_daily_DJFM_nonFltr_PC2.nc")) as f:
            pc2 = f.variables["PC2"][:]
        self.targets = np.stack([pc1, pc2], axis=-1)

        # Scan all files to create a full map from a global index to a file location.
        full_index_map = []
        self.file_paths = {}
        
        for lead_tm in self.lead_times:
            # Note the different variable names for OBS data
            self.file_paths[lead_tm] = {
                "tcwv": os.path.join(self.obs_dir, f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_tcwv_leadTm{lead_tm}.nc"),
                "olr": os.path.join(self.obs_dir, f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_olr_leadTm{lead_tm}.nc"),
                "u200": os.path.join(self.obs_dir, f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_u200_leadTm{lead_tm}.nc"),
                "u850": os.path.join(self.obs_dir, f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_u850_leadTm{lead_tm}.nc"),
                "trefht": os.path.join(self.obs_dir, f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_trefht_leadTm{lead_tm}.nc"),
                "LEADTM": os.path.join(self.time_dir, f"OBSleadTm{lead_tm}.nc")
            }
            with Dataset(self.file_paths[lead_tm]["tcwv"]) as f:
                num_in_file = f.variables["tcwv"].shape[0]
                for i in range(num_in_file):
                    target_idx = i
                    full_index_map.append((lead_tm, i, target_idx))

        # If a list of indices is provided, filter the map to represent only that subset.
        # Otherwise, use the full map.
        if indices is not None:
            self.index_map = [full_index_map[i] for i in indices]
        else:
            self.index_map = full_index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        lead_tm, local_idx, target_idx = self.index_map[idx]
        
        paths = self.file_paths[lead_tm]
        with Dataset(paths["tcwv"]) as f: tcwv = f.variables["tcwv"][local_idx, :, :]
        with Dataset(paths["olr"]) as f: olr = f.variables["olr"][local_idx, :, :]
        with Dataset(paths["u200"]) as f: u200 = f.variables["u200"][local_idx, :, :]
        with Dataset(paths["u850"]) as f: u850 = f.variables["u850"][local_idx, :, :]
        with Dataset(paths["trefht"]) as f: trefht = f.variables["trefht"][local_idx, :, :]
        with Dataset(paths["LEADTM"]) as f: leadtm = f.variables["LEADTM"][local_idx, :, :]
            
        data_sample = np.stack([tcwv, olr, u200, u850, trefht, leadtm], axis=-1)
        data_sample = data_sample.transpose(2, 0, 1)
        target_sample = self.targets[target_idx]
        
        x = torch.tensor(data_sample, dtype=torch.float32)
        y = torch.tensor(target_sample, dtype=torch.float32)
        
        return x, y
        
# General setup
lat = 30
lon = 180
var = 6
batch_size = 121*5

# Calculate the total number of samples without loading the data
# We check the size of the first lead time's PC file and multiply by the number of lead times
with Dataset(obs_directory + "CML2025_Step0C_OBS_remapped_90x180_daily_DJFM_nonFltr_PC1.nc") as f:
    num_samples_per_lead = len(f.variables["PC1"][:])
total_samples = num_samples_per_lead * len(lead_times_to_load)
print(f"Total OBS samples across all lead times: {total_samples}")

all_indices = np.arange(total_samples)

# --- Cross-validation splits (3:3:4) ---
test_ts = []
val_ts = []

# --- Round 1 ---
j = 1
pct = 1528
test_indices_1 = all_indices[j*pct:(j+1)*pct]
train_indices_1 = np.delete(all_indices, np.s_[j*pct:(j+1)*pct])

# Create a dataset representing only the training data for this fold
train_val_dataset_1 = OBSDiskDataset(obs_directory, time_dir, lead_times_to_load, indices=train_indices_1)
test_dataset_1 = OBSDiskDataset(obs_directory, time_dir, lead_times_to_load, indices=test_indices_1)

# Split training data into training and validation with Subset
N = len(train_val_dataset_1)
train_len = int(N * 0.8)
train_dataset_1 = torch.utils.data.Subset(train_val_dataset_1, list(range(0, train_len)))
val_dataset_1 = torch.utils.data.Subset(train_val_dataset_1, list(range(train_len, N)))

train_dataloader1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
val_dataloader1 = torch.utils.data.DataLoader(val_dataset_1, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test_dataloader1 = torch.utils.data.DataLoader(test_dataset_1, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
test_ts.append(len(test_dataset_1))
val_ts.append(len(val_dataset_1))

# --- Round 2 ---
j = 2
pct = 1528
test_indices_2 = all_indices[j*pct:]
train_indices_2 = np.delete(all_indices, np.s_[j*pct:])

train_val_dataset_2 = OBSDiskDataset(obs_directory, time_dir, lead_times_to_load, indices=train_indices_2)
test_dataset_2 = OBSDiskDataset(obs_directory, time_dir, lead_times_to_load, indices=test_indices_2)

N = len(train_val_dataset_2)
train_len = int(N * 0.8)
train_dataset_2 = torch.utils.data.Subset(train_val_dataset_2, list(range(0, train_len)))
val_dataset_2 = torch.utils.data.Subset(train_val_dataset_2, list(range(train_len, N)))

train_dataloader2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
val_dataloader2 = torch.utils.data.DataLoader(val_dataset_2, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test_dataloader2 = torch.utils.data.DataLoader(test_dataset_2, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
test_ts.append(len(test_dataset_2))
val_ts.append(len(val_dataset_2))

# --- Round 3 ---
j = 1
pct = 2036
test_indices_3 = all_indices[j*pct:(j+1)*pct]
train_indices_3 = np.delete(all_indices, np.s_[j*pct:(j+1)*pct])

train_val_dataset_3 = OBSDiskDataset(obs_directory, time_dir, lead_times_to_load, indices=train_indices_3)
test_dataset_3 = OBSDiskDataset(obs_directory, time_dir, lead_times_to_load, indices=test_indices_3)

N = len(train_val_dataset_3)
train_len = int(N * 0.8)
train_dataset_3 = torch.utils.data.Subset(train_val_dataset_3, list(range(0, train_len)))
val_dataset_3 = torch.utils.data.Subset(train_val_dataset_3, list(range(train_len, N)))

train_dataloader3 = torch.utils.data.DataLoader(train_dataset_3, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
val_dataloader3 = torch.utils.data.DataLoader(val_dataset_3, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test_dataloader3 = torch.utils.data.DataLoader(test_dataset_3, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
test_ts.append(len(test_dataset_3))
val_ts.append(len(val_dataset_3))

# --- Final Dataloader Lists ---
train_val_rounds = [[train_dataloader1, val_dataloader1], [train_dataloader2, val_dataloader2], [train_dataloader3, val_dataloader3]]
train_val_test_rounds = [[train_dataloader1, val_dataloader1, test_dataloader1], [train_dataloader2, val_dataloader2, test_dataloader2], [train_dataloader3, val_dataloader3, test_dataloader3]]

# Transfer Learning Training
lr = 1e-4
epochs = 300
lr_warmup_length = 5
compiled_train_losses = []
compiled_val_losses = []
for index, round in enumerate(train_val_rounds):
    train_losses = []
    val_losses = []
    train_dataloader = round[0]
    val_dataloader = round[1]
    model = torch.load(mdl_model_path, weights_only=False).to(device)
    model.pos_embedding.requires_grad = False
    for param in model.to_patch_embedding.parameters():
        param.requires_grad = False
    for param in model.transformer.norm.parameters():
        param.requires_grad = False
    for block in model.transformer.layers[:]:
        for param in block.parameters():
            param.requires_grad = False
    for param in model.mlp_head.parameters():
        param.requires_grad = True
    model.cls_token.requires_grad = False
    batches_per_dataset = len(train_dataloader)
    print(f"batches per dataset: {batches_per_dataset}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.0001, factor=0.5, mode='min')
    loss_fn = torch.nn.MSELoss()
    class EarlyStopping:
        def __init__(self, threshold, patience):
            self.threshold = threshold
            self.patience = patience
            self.patience_count = 0
        def check_early_stop(self, val_loss_history):
            if len(val_loss_history) < self.patience:
                return False
            if min(val_loss_history[:-1]) - val_loss_history[-1] < self.threshold:
                self.patience_count += 1
                if self.patience_count >= self.patience:
                    return True
            else:
                self.patience_count = 0
                return False
    earlystopping = EarlyStopping(threshold=0.0001, patience=10)
    best_validation_loss = float("inf")
    best_model = None
    model.eval()
    with torch.no_grad():
        total_train_loss = 0.0
        for batch_data, batch_target in train_dataloader:
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            y = model(batch_data)
            loss = loss_fn(y, batch_target)
            total_train_loss += loss.item() * batch_data.size(0)
        avg_train_loss = total_train_loss / len(train_dataloader.dataset)
        total_val_loss = 0.0
        for batch_data, batch_target in val_dataloader:
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
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
            optimizer = torch.optim.AdamW(model.parameters(), lr=warmup_lr, weight_decay=0.01)
        if i == lr_warmup_length:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.0001, factor=0.5, mode='min')
        for batch_data, batch_target in train_dataloader:
            batch_data = batch_data.to(device)
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
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_target in val_dataloader:
                batch_data = batch_data.to(device)
                batch_target = batch_target.to(device)        
                pred_y = model(batch_data)
                loss = loss_fn(pred_y, batch_target)
                total_val_loss += loss.item() * batch_data.size(0)
        avg_val_loss = total_val_loss / len(val_dataloader.dataset)
        if i >= lr_warmup_length:
            scheduler.step(avg_val_loss)
        print(f"Epoch: {i + 1}, Avg Train Loss: {avg_train_loss}, Avg Val Loss: {avg_val_loss}, LR: {scheduler.get_last_lr()}")
        if avg_val_loss < best_validation_loss:
            best_validation_loss = avg_val_loss
            best_model = copy.deepcopy(model)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        if earlystopping.check_early_stop(val_losses):
            break
    model = best_model
    model.eval()
    torch.save(model, obs_out_dir + f"ViTTIMJO_Andrew_OBS_leadTm{leadTm}_ensm{seed_num}_round{index + 1}.pth")
    compiled_train_losses.append(train_losses)
    compiled_val_losses.append(val_losses)

print("\nCombined MDL and OBS Transfer Learning Training Completed.")

import os
import torch
import numpy as np
from netCDF4 import Dataset

# ====================================================================
# Helper Function to Get Predictions
# ====================================================================

def get_all_preds_and_targets(model, dataloader, device):
    """
    Runs inference using the provided model and dataloader, returning
    all predictions and corresponding targets as single tensors.
    """
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_targets = []
    with torch.no_grad():  # Disable gradient calculations for inference
        for features, targets in dataloader:
            # Move input features to the correct device (e.g., GPU)
            features = features.to(device)
            
            # Get model prediction
            pred = model(features)
            
            # Append results to lists, moving them to the CPU to free up GPU memory
            all_preds.append(pred.detach().cpu())
            all_targets.append(targets.cpu()) # Targets are likely already on CPU
            
    # Concatenate all collected batches into single, large tensors
    return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)


# ====================================================================
# Section 3: Compute Final Predictions and Save to NetCDF (Memory-Efficient)
# ====================================================================

# This section assumes 'train_val_test_rounds' was created in the refactored Section 2.
# It should be a list of lists, e.g., [[train_dl_1, val_dl_1, test_dl_1], [train_dl_2, ...]]
# Also assumes 'device', 'obs_out_dir', 'bcor_directory', 'leadTm', and 'seed_num' are defined.

print("\n--- Starting Section 3: Final Prediction and Saving ---")

for index, round_dataloaders in enumerate(train_val_test_rounds):
    round_num = index + 1
    print(f"\nProcessing Cross-Validation Round {round_num}...")
    
    # Unpack the dataloaders for this round
    train_dataloader, val_dataloader, test_dataloader = round_dataloaders
    
    # --- Load the best model for this round ---
    model_path = os.path.join(obs_out_dir, f"ViTTIMJO_Andrew_OBS_leadTm{leadTm}_ensm{seed_num}_round{round_num}.pth")
    
    if not os.path.exists(model_path):
        print(f"WARNING: Model file not found, skipping round {round_num}. Path: {model_path}")
        continue
        
    print(f"  Loading model: {model_path}")
    model = torch.load(model_path, map_location=device)
    model.to(device)
    
    # --- Get predictions for all datasets in this round using the helper function ---
    print("  Generating predictions for Train, Validation, and Test sets...")
    train_preds, train_targets = get_all_preds_and_targets(model, train_dataloader, device)
    val_preds,   val_targets   = get_all_preds_and_targets(model, val_dataloader, device)
    test_preds,  test_targets  = get_all_preds_and_targets(model, test_dataloader, device)
    print("  Prediction generation complete.")

    # --- Write results to NetCDF files ---
    print("  Writing results to NetCDF files...")
    # The number of samples is now correctly retrieved from the length of the dataset in each dataloader
    datasets_to_save = [
        ("Train", train_preds, train_targets, len(train_dataloader.dataset)),
        ("Val",   val_preds,   val_targets,   len(val_dataloader.dataset)),
        ("Test",  test_preds,  test_targets,  len(test_dataloader.dataset))
    ]
    
    for typ, preds, tgts, num_samples in datasets_to_save:
        fname = os.path.join(bcor_directory, f"ViTTIMJO_Andrew_OBS_leadTm{leadTm}_ensm{seed_num}_rund{round_num}_Prdct_{typ}.nc")
        
        # Ensure the output directory exists before writing
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        
        if os.path.exists(fname):
            os.remove(fname)
            
        with Dataset(fname, "w", format="NETCDF4") as nc_file:
            # Create dimensions
            nc_file.createDimension("time", num_samples)
            nc_file.createDimension("var", 2)
            
            # Define variable names and save data
            pred_var = nc_file.createVariable(f"mjo{typ}Pred", "f4", ("time", "var",))
            pred_var[:] = preds.numpy()
            
            tgt_var = nc_file.createVariable(f"mjo{typ}Tagt", "f4", ("time", "var",))
            tgt_var[:] = tgts.numpy()
            
        print(f"    - Saved: {fname}")

print("\nPrediction computation and NetCDF writing completed.")