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

leadTm = 15

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

tot = 43076
lat = 30
lon = 180
var = 6

#Load MDL data
in1 = Dataset(mdl_directory + f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_TMQ_leadTm{leadTm}.nc")
in2 = Dataset(mdl_directory + f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_FLUT_leadTm{leadTm}.nc")
in3 = Dataset(mdl_directory + f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_U200_leadTm{leadTm}.nc")
in4 = Dataset(mdl_directory + f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_U850_leadTm{leadTm}.nc")
in5 = Dataset(mdl_directory + f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_TREFHT_leadTm{leadTm}.nc")
in6 = Dataset(f"Data/Time/MDLleadTm{leadTm}.nc")  # Lead time constant
out1 = Dataset(mdl_directory + "CML2025_Step0C_MDL_remapped_90x180_daily_DJFM_nonFltr_PC1.nc")
out2 = Dataset(mdl_directory + "CML2025_Step0C_MDL_remapped_90x180_daily_DJFM_nonFltr_PC2.nc")
if deBug:
    print("MDL in1  = ", in1,  "\n")
    print("MDL out1 = ", out1, "\n")

# Aggregate MDL data
mdl_data = np.zeros((tot, lat, lon, var), dtype=np.float32)
mdl_target = np.zeros((tot, 2), dtype=np.float32)
mdl_data[...,0] = in1.variables["TMQ"][:,:,:]
mdl_data[...,1] = in2.variables["FLUT"][:,:,:]
mdl_data[...,2] = in3.variables["U200"][:,:,:]
mdl_data[...,3] = in4.variables["U850"][:,:,:]
mdl_data[...,4] = in5.variables["TREFHT"][:,:,:]
mdl_data[...,5] = in6.variables["LEADTM"][:,:,:]  # Lead time constant
mdl_target[:,0] = out1.variables["PC1"][:]
mdl_target[:,1] = out2.variables["PC2"][:]
if deBug:
    print("MDL data.shape   = ", mdl_data.shape,   "\n")
    print("MDL target.shape = ", mdl_target.shape, "\n")

mdl_data = mdl_data.transpose(0,3,1,2)
print(mdl_data.shape)


class MDLDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        # Store as numpy arrays; convert to tensor in __getitem__ for memory efficiency
        self.features = features
        self.targets = targets
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32).to(device)
        y = torch.tensor(self.targets[idx], dtype=torch.float32).to(device)
        return x, y

mdl_dataset = MDLDataset(mdl_data, mdl_target)
mdl_total_samples = len(mdl_dataset)
mdl_train_size = int(0.8 * mdl_total_samples) - int(0.8 * mdl_total_samples) % 121
mdl_test_size = mdl_total_samples - mdl_train_size
mdl_train_dataset, mdl_test_dataset = torch.utils.data.random_split(mdl_dataset, [mdl_train_size, mdl_test_size])
batch_size = 121
train_dataloader = torch.utils.data.DataLoader(mdl_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
test_dataloader = torch.utils.data.DataLoader(mdl_test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

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
        y = model(batch_data)
        loss = loss_fn(y, batch_target)
        total_train_loss += loss.item() * batch_data.size(0)
    avg_train_loss = total_train_loss / len(train_dataloader.dataset)
    total_test_loss = 0.0
    for batch_data, batch_target in test_dataloader:
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

del mdl_data, mdl_target, mdl_dataset, train_dataloader, test_dataloader
import gc; gc.collect()

# --------------------
# Section 2: OBS Transfer Learning (as in ViT Train 2.py)
# --------------------

tot = 5092
lat = 30
lon = 180
var = 6

batch_size = 121

in1 = Dataset(obs_directory + f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_tcwv_leadTm{leadTm}.nc")
in2 = Dataset(obs_directory + f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_olr_leadTm{leadTm}.nc")
in3 = Dataset(obs_directory + f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_u200_leadTm{leadTm}.nc")
in4 = Dataset(obs_directory + f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_u850_leadTm{leadTm}.nc")
in5 = Dataset(obs_directory + f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_trefht_leadTm{leadTm}.nc")
in6 = Dataset(time_dir + f"OBSleadTm{leadTm}.nc")  # Lead time constant
out1 = Dataset(obs_directory + "CML2025_Step0C_OBS_remapped_90x180_daily_DJFM_nonFltr_PC1.nc")
out2 = Dataset(obs_directory + "CML2025_Step0C_OBS_remapped_90x180_daily_DJFM_nonFltr_PC2.nc")
if deBug:
    print("OBS in1  = ", in1,  "\n")
    print("OBS out1 = ", out1, "\n")

obs_data = np.zeros((tot, lat, lon, var), dtype=np.float32)
obs_target = np.zeros((tot, 2), dtype=np.float32)
obs_data[...,0] = in1.variables["tcwv"][:,:,:]
obs_data[...,1] = in2.variables["olr"][:,:,:]
obs_data[...,2] = in3.variables["u200"][:,:,:]
obs_data[...,3] = in4.variables["u850"][:,:,:]
obs_data[...,4] = in5.variables["trefht"][:,:,:]
obs_data[...,5] = in6.variables["LEADTM"][:,:,:]
obs_target[:,0] = out1.variables["PC1"][:]
obs_target[:,1] = out2.variables["PC2"][:]
if deBug:
    print("OBS data.shape   = ", obs_data.shape,   "\n")
    print("OBS target.shape = ", obs_target.shape, "\n")
obs_data = obs_data.transpose(0,3,1,2)
print(obs_data.shape)

class OBSDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        # Store as numpy arrays; convert to tensor in __getitem__ for memory efficiency
        self.features = features
        self.targets = targets
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32).to(device)
        y = torch.tensor(self.targets[idx], dtype=torch.float32).to(device)
        return x, y

# Cross-validation splits (3:3:4)
test_ts = []
val_ts = []
# Round 1
j = 1
pct = 1528
train_x_1 = np.delete(obs_data,   np.s_[j*pct:(j+1)*pct], axis=0)
train_y_1 = np.delete(obs_target, np.s_[j*pct:(j+1)*pct], axis=0)
test_x_1  = obs_data  [j*pct:(j+1)*pct]
test_y_1  = obs_target[j*pct:(j+1)*pct]
test_t_1  = np.shape(test_y_1)[0]
train_data_1 = OBSDataset(train_x_1, train_y_1)
N = len(train_data_1)
train_len = int(N * 0.8)
train_dataset_1 = torch.utils.data.Subset(train_data_1, list(range(0, train_len)))
val_dataset_1 = torch.utils.data.Subset(train_data_1, list(range(train_len, N)))
test_dataset_1 = OBSDataset(test_x_1, test_y_1)
train_dataloader1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True, pin_memory=False)
val_dataloader1 = torch.utils.data.DataLoader(val_dataset_1, batch_size=batch_size, shuffle=True, pin_memory=False)
test_dataloader1 = torch.utils.data.DataLoader(test_dataset_1, batch_size=batch_size, shuffle=False, pin_memory=False)
test_ts.append(test_t_1)
val_ts.append(len(val_dataloader1.dataset))
# Round 2
j = 2
pct = 1528
train_x_2 = np.delete(obs_data,   np.s_[j*pct:], axis=0)
train_y_2 = np.delete(obs_target, np.s_[j*pct:], axis=0)
test_x_2  = obs_data  [j*pct:]
test_y_2  = obs_target[j*pct:]
test_t_2  = np.shape(test_y_2)[0]
train_data_2 = OBSDataset(train_x_2, train_y_2)
N = len(train_data_2)
train_len = int(N * 0.8)
train_dataset_2 = torch.utils.data.Subset(train_data_2, list(range(0, train_len)))
val_dataset_2 = torch.utils.data.Subset(train_data_2, list(range(train_len, N)))
test_dataset_2 = OBSDataset(test_x_2, test_y_2)
train_dataloader2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True, pin_memory=False)
val_dataloader2 = torch.utils.data.DataLoader(val_dataset_2, batch_size=batch_size, shuffle=True, pin_memory=False)
test_dataloader2 = torch.utils.data.DataLoader(test_dataset_2, batch_size=batch_size, shuffle=False, pin_memory=False)
test_ts.append(test_t_2)
val_ts.append(len(val_dataloader2.dataset))
# Round 3
j = 1
pct = 2036
train_x_3 = np.delete(obs_data,   np.s_[j*pct:(j+1)*pct], axis=0)
train_y_3 = np.delete(obs_target, np.s_[j*pct:(j+1)*pct], axis=0)
test_x_3  = obs_data  [j*pct:(j+1)*pct]
test_y_3  = obs_target[j*pct:(j+1)*pct]
test_t_3  = np.shape(test_y_3)[0]
train_data_3 = OBSDataset(train_x_3, train_y_3)
N = len(train_data_3)
train_len = int(N * 0.8)
train_dataset_3 = torch.utils.data.Subset(train_data_3, list(range(0, train_len)))
val_dataset_3 = torch.utils.data.Subset(train_data_3, list(range(train_len, N)))
test_dataset_3 = OBSDataset(test_x_3, test_y_3)
train_dataloader3 = torch.utils.data.DataLoader(train_dataset_3, batch_size=batch_size, shuffle=True, pin_memory=False)
val_dataloader3 = torch.utils.data.DataLoader(val_dataset_3, batch_size=batch_size, shuffle=True, pin_memory=False)
test_dataloader3 = torch.utils.data.DataLoader(test_dataset_3, batch_size=batch_size, shuffle=False, pin_memory=False)
test_ts.append(test_t_3)
val_ts.append(len(val_dataloader3.dataset))

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
            y = model(batch_data)
            loss = loss_fn(y, batch_target)
            total_train_loss += loss.item() * batch_data.size(0)
        avg_train_loss = total_train_loss / len(train_dataloader.dataset)
        total_val_loss = 0.0
        for batch_data, batch_target in val_dataloader:
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

# --------------------
# Section 3: Compute Final Predictions and Save to NetCDF
# --------------------

from netCDF4 import Dataset
import numpy as np
import torch
import os

test_ts = []
val_ts = []
# Cross-validation splits (3:3:4) for obs_data (already defined above)
# Round 1
j = 1
pct = 1528
train_x_1 = np.delete(obs_data,   np.s_[j*pct:(j+1)*pct], axis=0)
train_y_1 = np.delete(obs_target, np.s_[j*pct:(j+1)*pct], axis=0)
test_x_1  = obs_data  [j*pct:(j+1)*pct]
test_y_1  = obs_target[j*pct:(j+1)*pct]
test_t_1  = np.shape(test_y_1)[0]
train_data_1 = OBSDataset(train_x_1, train_y_1)
N = len(train_data_1)
train_len = int(N * 0.8)
train_dataset_1 = torch.utils.data.Subset(train_data_1, list(range(0, train_len)))
val_dataset_1 = torch.utils.data.Subset(train_data_1, list(range(train_len, N)))
test_dataset_1 = OBSDataset(test_x_1, test_y_1)
train_dataloader1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True, pin_memory=False)
val_dataloader1 = torch.utils.data.DataLoader(val_dataset_1, batch_size=batch_size, shuffle=True, pin_memory=False)
test_dataloader1 = torch.utils.data.DataLoader(test_dataset_1, batch_size=batch_size, shuffle=False, pin_memory=False)
test_ts.append(test_t_1)
val_ts.append(len(val_dataloader1.dataset))
# Round 2
j = 2
pct = 1528
train_x_2 = np.delete(obs_data,   np.s_[j*pct:], axis=0)
train_y_2 = np.delete(obs_target, np.s_[j*pct:], axis=0)
test_x_2  = obs_data  [j*pct:]
test_y_2  = obs_target[j*pct:]
test_t_2  = np.shape(test_y_2)[0]
train_data_2 = OBSDataset(train_x_2, train_y_2)
N = len(train_data_2)
train_len = int(N * 0.8)
train_dataset_2 = torch.utils.data.Subset(train_data_2, list(range(0, train_len)))
val_dataset_2 = torch.utils.data.Subset(train_data_2, list(range(train_len, N)))
test_dataset_2 = OBSDataset(test_x_2, test_y_2)
train_dataloader2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True, pin_memory=False)
val_dataloader2 = torch.utils.data.DataLoader(val_dataset_2, batch_size=batch_size, shuffle=True, pin_memory=False)
test_dataloader2 = torch.utils.data.DataLoader(test_dataset_2, batch_size=batch_size, shuffle=False, pin_memory=False)
test_ts.append(test_t_2)
val_ts.append(len(val_dataloader2.dataset))
# Round 3
j = 1
pct = 2036
train_x_3 = np.delete(obs_data,   np.s_[j*pct:(j+1)*pct], axis=0)
train_y_3 = np.delete(obs_target, np.s_[j*pct:(j+1)*pct], axis=0)
test_x_3  = obs_data  [j*pct:(j+1)*pct]
test_y_3  = obs_target[j*pct:(j+1)*pct]
test_t_3  = np.shape(test_y_3)[0]
train_data_3 = OBSDataset(train_x_3, train_y_3)
N = len(train_data_3)
train_len = int(N * 0.8)
train_dataset_3 = torch.utils.data.Subset(train_data_3, list(range(0, train_len)))
val_dataset_3 = torch.utils.data.Subset(train_data_3, list(range(train_len, N)))
test_dataset_3 = OBSDataset(test_x_3, test_y_3)
train_dataloader3 = torch.utils.data.DataLoader(train_dataset_3, batch_size=batch_size, shuffle=True, pin_memory=False)
val_dataloader3 = torch.utils.data.DataLoader(val_dataset_3, batch_size=batch_size, shuffle=True, pin_memory=False)
test_dataloader3 = torch.utils.data.DataLoader(test_dataset_3, batch_size=batch_size, shuffle=False, pin_memory=False)
test_ts.append(test_t_3)
val_ts.append(len(val_dataloader3.dataset))

train_val_test_rounds = [
    [train_dataloader1, val_dataloader1, test_dataloader1],
    [train_dataloader2, val_dataloader2, test_dataloader2],
    [train_dataloader3, val_dataloader3, test_dataloader3]
]

for index, round in enumerate(train_val_test_rounds):
    print(f"Round {index+1}")
    train_dataloader = round[0]
    val_dataloader = round[1]
    test_dataloader = round[2]
    # Load best model for this round
    model = torch.load(obs_out_dir + f"ViTTIMJO_Andrew_OBS_leadTm{leadTm}_ensm{seed_num}_round{index + 1}.pth", weights_only=False).to(device)
    model.eval()
    all_train_preds = []
    all_train_targets = []
    all_val_preds = []
    all_val_targets = []
    all_test_preds = []
    all_test_targets = []
    with torch.no_grad():
        for batch_data, batch_target in train_dataloader:
            y = model(batch_data)
            all_train_preds.append(y.detach().cpu())
            all_train_targets.append(batch_target.detach().cpu())
        for batch_data, batch_target in val_dataloader:
            y = model(batch_data)
            all_val_preds.append(y.detach().cpu())
            all_val_targets.append(batch_target.detach().cpu())
        for batch_data, batch_target in test_dataloader:
            pred_y = model(batch_data)
            all_test_preds.append(pred_y.detach().cpu())
            all_test_targets.append(batch_target.detach().cpu())
    all_train_preds = torch.cat(all_train_preds, dim=0)
    all_train_targets = torch.cat(all_train_targets, dim=0)
    all_val_preds = torch.cat(all_val_preds, dim=0)
    all_val_targets = torch.cat(all_val_targets, dim=0)
    all_test_preds = torch.cat(all_test_preds, dim=0)
    all_test_targets = torch.cat(all_test_targets, dim=0)
    # Write to NetCDF
    for typ, preds, tgts, ts in zip(
        ["Train", "Val", "Test"],
        [all_train_preds, all_val_preds, all_test_preds],
        [all_train_targets, all_val_targets, all_test_targets],
        [tot - val_ts[index] - test_ts[index], val_ts[index], test_ts[index]]
    ):
        fname = bcor_directory + f"ViTTIMJO_Andrew_OBS_leadTm{leadTm}_ensm{seed_num}_rund{index+1}_Prdct_{typ}.nc"
        if os.path.exists(fname):
            os.remove(fname)
        create_NC = Dataset(fname, "w", format="NETCDF4")
        create_NC.createDimension("time", ts)
        create_NC.createDimension("var", 2)
        if typ == "Train":
            create_NC.createVariable("mjoTrainPred", "f4", ("time","var",))[:] = np.array(preds.cpu().numpy(), dtype=np.float32)
            create_NC.createVariable("mjoTrainTagt", "f4", ("time","var",))[:] = np.array(tgts.cpu().numpy(), dtype=np.float32)
        elif typ == "Val":
            create_NC.createVariable("mjoValPred", "f4", ("time","var",))[:] = np.array(preds.cpu().numpy(), dtype=np.float32)
            create_NC.createVariable("mjoValTagt", "f4", ("time","var",))[:] = np.array(tgts.cpu().numpy(), dtype=np.float32)
        elif typ == "Test":
            create_NC.createVariable("mjoTestPred", "f4", ("time","var",))[:] = np.array(preds.cpu().numpy(), dtype=np.float32)
            create_NC.createVariable("mjoTestTagt", "f4", ("time","var",))[:] = np.array(tgts.cpu().numpy(), dtype=np.float32)
        create_NC.close()

print("\nPrediction computation and NetCDF writing completed.")
