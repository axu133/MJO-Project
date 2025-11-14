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
    
    # --------------------
    # Section 2: OBS Transfer Learning (as in ViT Train 2.py)
    # --------------------

    # General setup

    mdl_model_path = mdl_out_dir + f"ViTTIMJO_FiLM_Andrew_MDL_leadTm{model_leadTms}_ensm{seed_num}.pth"

    lat = 30
    lon = 180
    var = 5
    batch_size = 121

    # Calculate the total number of samples without loading the data
    # We check the size of the first lead time's PC file and multiply by the number of lead times
    with Dataset(obs_directory + "CML2025_Step0C_OBS_remapped_90x180_daily_DJFM_nonFltr_PC1.nc") as f:
        total_samples = len(f.variables["PC1"][:])
    print(f"Total OBS samples across all lead times: {total_samples}")

    indices = np.arange(total_samples)

    # For each lead time, split indices into 3 folds
    n = len(indices)
    fold1 = indices[:n//3]
    fold2 = indices[n//3:2*n//3]
    fold3 = indices[2*n//3:]
    folds = [fold1, fold2, fold3] # Each will be a list of indices for fold 1, 2, 3

    # For each round, use one fold as test, one as val, one as train (rotate)
    train_val_rounds = []
    train_val_test_rounds = []
    test_ts = []
    val_ts = []

    for i in range(3):
        test_indices = np.array(folds[i])
        val_indices = np.array(folds[(i+1)%3])
        train_indices = np.array(folds[(i+2)%3])

        # Create datasets
        train_dataset = OBSDataset(obs_directory, time_dir, lead_time_width, indices=train_indices)
        val_dataset = OBSDataset(obs_directory, time_dir, lead_time_width, indices=val_indices)
        test_dataset = OBSDataset(obs_directory, time_dir, lead_time_width, indices=test_indices)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
        test_ts.append(len(test_dataset))
        val_ts.append(len(val_dataset))
        train_val_rounds.append([train_dataloader, val_dataloader])
        train_val_test_rounds.append([train_dataloader, val_dataloader, test_dataloader])
        
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
        for param in model.decoder_head.parameters():
            param.requires_grad = True
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
        torch.save(model, obs_out_dir + f"ViTTIMJO_FiLM_Andrew_OBS_leadTm{model_leadTms}_ensm{seed_num}_round{index + 1}.pth")
        compiled_train_losses.append(train_losses)
        compiled_val_losses.append(val_losses)

    print("\nCombined MDL and OBS Transfer Learning Training Completed.")