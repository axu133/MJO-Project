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
lead_times_to_load = range(10,31)

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
    """
    Loads the entire MDL dataset into NumPy arrays in RAM.
    Tensor conversion happens on-the-fly in __getitem__.
    """
    def __init__(self, mdl_dir, time_dir, lead_times):
        print("Loading MDL dataset into memory (NumPy arrays)...")
        
        # Load Targets into a NumPy array
        with Dataset(os.path.join(mdl_dir, "CML2025_Step0C_MDL_remapped_90x180_daily_DJFM_nonFltr_PC1.nc")) as f:
            pc1 = f.variables["PC1"][:]
        with Dataset(os.path.join(mdl_dir, "CML2025_Step0C_MDL_remapped_90x180_daily_DJFM_nonFltr_PC2.nc")) as f:
            pc2 = f.variables["PC2"][:]
        self.targets_np = np.stack([pc1, pc2], axis=-1)

        # Load Features and Conditioning into NumPy arrays 
        all_features = []
        all_conditioning = []
        target_indices_list = []
        variable_names = ["TMQ", "FLUT", "U200", "U850", "TREFHT"]
        
        for lead_tm in lead_times:
            # Load all variables for the current lead time
            lead_time_vars = []
            for var_name in variable_names:
                file_path = os.path.join(mdl_dir, f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_{var_name}_leadTm{lead_tm}.nc")
                with Dataset(file_path) as f:
                    lead_time_vars.append(f.variables[var_name][:])
            
            # Stack variables along a new channel dimension -> (samples, vars, lat, lon)
            stacked_features = np.stack(lead_time_vars, axis=1)
            all_features.append(stacked_features)
            
            num_samples = stacked_features.shape[0]

            for i in range(num_samples):
                target_indices_list.append(i)

            conditioning_data = np.full((num_samples, 1), lead_tm / 40.0)
            all_conditioning.append(conditioning_data)

        # Concatenate all lead times into single large NumPy arrays
        self.features_np = np.concatenate(all_features, axis=0)
        self.conditioning_np = np.concatenate(all_conditioning, axis=0)
        self.target_indices = np.array(target_indices_list)
        
        print(f"MDL dataset loaded. Total samples: {len(self.features_np)}")

    def __len__(self):
        return len(self.features_np)

    def __getitem__(self, idx):
        # Retrieve the data as NumPy arrays
        target_idx = self.target_indices[idx]
        
        features = self.features_np[idx]
        conditioning = self.conditioning_np[idx]
        targets = self.targets_np[target_idx]
        
        # Convert only this specific sample to a tensor before returning
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(conditioning, dtype=torch.float32),
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

class OBSDataset(torch.utils.data.Dataset):
    """
    Loads the OBS dataset into NumPy arrays in RAM and can represent a subset.
    Tensor conversion happens on-the-fly in __getitem__.
    """
    def __init__(self, obs_dir, time_dir, lead_times, indices=None):
        print("Loading OBS dataset into memory (NumPy arrays)...")
        
        # --- Load all data into full NumPy arrays first ---
        with Dataset(os.path.join(obs_dir, "CML2025_Step0C_OBS_remapped_90x180_daily_DJFM_nonFltr_PC1.nc")) as f:
            pc1 = f.variables["PC1"][:]
        with Dataset(os.path.join(obs_dir, "CML2025_Step0C_OBS_remapped_90x180_daily_DJFM_nonFltr_PC2.nc")) as f:
            pc2 = f.variables["PC2"][:]
        self.targets_np = np.stack([pc1, pc2], axis=-1)

        all_features = []
        all_conditioning = []
        target_indices_list = []
        variable_names = ["tcwv", "olr", "u200", "u850", "trefht"]

        for lead_tm in lead_times:
            lead_time_vars = []
            for var_name in variable_names:
                file_path = os.path.join(obs_dir, f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_{var_name}_leadTm{lead_tm}.nc")
                with Dataset(file_path) as f:
                    lead_time_vars.append(f.variables[var_name][:])
            
            stacked_features = np.stack(lead_time_vars, axis=1)
            all_features.append(stacked_features)
            
            num_samples = stacked_features.shape[0]

            for i in range(num_samples):
                target_indices_list.append(i)

            conditioning_data = np.full((num_samples, 1), lead_tm / 40.0)
            all_conditioning.append(conditioning_data)

        features_full_np = np.concatenate(all_features, axis=0)
        conditioning_full_np = np.concatenate(all_conditioning, axis=0)
        target_indices_full_np = np.array(target_indices_list)

        # --- Filter by indices if provided to create the final dataset ---
        if indices is not None:
            self.features_np = features_full_np[indices]
            self.conditioning_np = conditioning_full_np[indices]
            self.target_indices = target_indices_full_np[indices]
            print(f"OBS dataset subset loaded. Total samples: {len(self.features_np)}")
        else:
            self.features_np = features_full_np
            self.conditioning_np = conditioning_full_np
            self.target_indices = target_indices_full_np
            print(f"OBS dataset loaded. Total samples: {len(self.features_np)}")

    def __len__(self):
        return len(self.features_np)

    def __getitem__(self, idx):
        # Retrieve the data as NumPy arrays
        target_idx = self.target_indices[idx]
        
        features = self.features_np[idx]
        conditioning = self.conditioning_np[idx]
        targets = self.targets_np[target_idx]
        
        # Convert only this specific sample to a tensor before returning
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(conditioning, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32)
        )

def get_all_preds_and_targets(model, dataloader, device):
    """
    Runs inference using the provided model and dataloader, returning
    all predictions and corresponding targets as single tensors.
    """
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_targets = []
    all_lead_times = []
    with torch.no_grad():  # Disable gradient calculations for inference
        for features, conditioning, targets in dataloader:
            # Move input features to the correct device (e.g., GPU)
            features = features.to(device)
            conditioning = conditioning.to(device)

            # Get model prediction
            pred = model(features, conditioning)

            lead_times = conditioning[:,0]
            
            # Append results to lists, moving them to the CPU to free up GPU memory
            all_preds.append(pred.detach().cpu())
            all_lead_times.append(lead_times.cpu())
            all_targets.append(targets.cpu()) # Targets are likely already on CPU
            
    # Concatenate all collected batches into single, large tensors
    preds_tensor = torch.cat(all_preds, dim=0)
    lead_times_tensor = torch.cat(all_lead_times, dim=0) * 40
    targets_tensor = torch.cat(all_targets, dim=0)

    return preds_tensor, lead_times_tensor, targets_tensor 

def save_to_netcdf(preds, tgts, lead_times, typ, bcor_directory, model_path):
    unique_lead_times = torch.unique(lead_times)  # Get unique lead times
    for lead_time in unique_lead_times:
        # Filter predictions and targets for the current lead time
        mask = lead_times == lead_time
        preds_filtered = preds[mask]
        tgts_filtered = tgts[mask]
        num_samples = preds_filtered.shape[0]

        # Extract base filename from model_path
        base_filename = os.path.basename(model_path).replace(".pth", "")
        
        # Create filename dynamically based on lead time
        fname = os.path.join(
            bcor_directory,
            f"{base_filename}_Prdct_{typ}_leadTm{int(lead_time.item())}.nc"
        )

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
            pred_var[:] = preds_filtered.numpy()

            tgt_var = nc_file.createVariable(f"mjo{typ}Tagt", "f4", ("time", "var",))
            tgt_var[:] = tgts_filtered.numpy()

        print(f"    - Saved: {fname}")

if __name__ == "__main__":
 
  # --------------------
  # Section 1: MDL Training (as in ViT Train.py)
  # --------------------
  
  # Create the memory-efficient dataset
  mdl_dataset = MDLDataset(mdl_dir=mdl_directory, time_dir=time_dir, lead_times=lead_times_to_load)

  # --- The rest of your script remains largely the same ---

  # Split into training and testing sets
  mdl_total_samples = len(mdl_dataset)
  mdl_train_size = int(0.8 * mdl_total_samples) - int(0.8 * mdl_total_samples) % 121
  mdl_test_size = mdl_total_samples - mdl_train_size
  mdl_train_dataset, mdl_test_dataset = torch.utils.data.random_split(mdl_dataset, [mdl_train_size, mdl_test_size])

  # Create DataLoaders
  batch_size = 121
  # Set num_workers > 0 to load data in parallel, which is highly recommended with this on-disk approach
  train_dataloader = torch.utils.data.DataLoader(mdl_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=4)
  test_dataloader = torch.utils.data.DataLoader(mdl_test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=4)

  # ViT Model
  model = ViT(
      image_size=180,
      patch_size=15,
      num_classes=2,
      dim=256,
      depth=6,
      heads=8,
      mlp_dim=512,
      channels=5,
      dropout=0.1,
      emb_dropout=0.1
  )

  lr = 0.0005
  epochs = 500
  model = model.to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.0001, factor=0.5, mode='min')
  #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
  loss_fn = torch.nn.MSELoss()

  earlystopping = EarlyStopping(threshold=0.0001, patience=10)

  # Initial evaluation
  train_losses = []
  test_losses = []
  with torch.no_grad():
      model.eval()
      total_train_loss = 0.0
      for batch_data, batch_conditioning, batch_target in train_dataloader:
          batch_data = batch_data.to(device)
          batch_conditioning = batch_conditioning.to(device)
          batch_target = batch_target.to(device)
          y = model(batch_data, batch_conditioning)
          loss = loss_fn(y, batch_target)
          total_train_loss += loss.item() * batch_data.size(0)
      avg_train_loss = total_train_loss / len(train_dataloader.dataset)
      total_test_loss = 0.0
      for batch_data, batch_conditioning, batch_target in test_dataloader:
          batch_data = batch_data.to(device)
          batch_conditioning = batch_conditioning.to(device)
          batch_target = batch_target.to(device)
          pred_y = model(batch_data, batch_conditioning)
          loss = loss_fn(pred_y, batch_target)
          total_test_loss += loss.item() * batch_data.size(0)
      avg_test_loss = total_test_loss / len(test_dataloader.dataset)
      print(f"Initial Performance of Untrained Model: \nAvg Train Loss: {avg_train_loss}, Avg Test Loss: {avg_test_loss}")
      train_losses.append(avg_train_loss)
      test_losses.append(avg_test_loss)

  for i in range(epochs):
      total_train_loss = 0.0
      for batch_data, batch_conditioning, batch_target in train_dataloader:
          batch_data = batch_data.to(device)
          batch_conditioning = batch_conditioning.to(device)
          batch_target = batch_target.to(device)
          model.train()
          y = model(batch_data, batch_conditioning)
          loss = loss_fn(y, batch_target)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          total_train_loss += loss.item() * batch_data.size(0)
      avg_train_loss = total_train_loss / len(train_dataloader.dataset)
      model.eval()
      total_test_loss = 0.0
      with torch.no_grad():
          for batch_data, batch_conditioning, batch_target in test_dataloader:
              batch_data = batch_data.to(device)
              batch_conditioning = batch_conditioning.to(device)
              batch_target = batch_target.to(device)
              pred_y = model(batch_data,  batch_conditioning)
              loss = loss_fn(pred_y, batch_target)
              total_test_loss += loss.item() * batch_data.size(0)
      avg_test_loss = total_test_loss / len(test_dataloader.dataset)
      train_losses.append(avg_train_loss)
      test_losses.append(avg_test_loss)
      scheduler.step(avg_test_loss)
      #scheduler.step()
      print(f"Epoch: {i + 1}, Avg Train Loss: {avg_train_loss}, Avg Test Loss: {avg_test_loss}, LR: {scheduler.get_last_lr()}")
      
      if earlystopping.check_early_stop(test_losses):
          break
    
    

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
      num_samples_per_lead = len(f.variables["PC1"][:])
  total_samples = num_samples_per_lead * len(lead_times_to_load)
  print(f"Total OBS samples across all lead times: {total_samples}")

  all_indices = np.arange(total_samples)

  # Calculate the total number of samples per lead time
  with Dataset(obs_directory + "CML2025_Step0C_OBS_remapped_90x180_daily_DJFM_nonFltr_PC1.nc") as f:
      num_samples_per_lead = len(f.variables["PC1"][:])
  total_samples = num_samples_per_lead * len(lead_times_to_load)
  print(f"Total OBS samples across all lead times: {total_samples}")
  
  # Stratified cross-validation split by lead time
  lead_time_indices = {}
  for i, lead_tm in enumerate(lead_times_to_load):
      start = i * num_samples_per_lead
      end = (i + 1) * num_samples_per_lead
      lead_time_indices[lead_tm] = np.arange(start, end)
  
  # For each lead time, split indices into 3 folds (approx 3:3:4)
  folds = [[], [], []]  # Each will be a list of indices for fold 1, 2, 3
  for lead_tm in lead_times_to_load:
      indices = lead_time_indices[lead_tm]
      n = len(indices)
      fold1 = indices[:n//3]
      fold2 = indices[n//3:2*n//3]
      fold3 = indices[2*n//3:]
      folds[0].extend(fold1)
      folds[1].extend(fold2)
      folds[2].extend(fold3)
  
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
      train_dataset = OBSDataset(obs_directory, time_dir, lead_times_to_load, indices=train_indices)
      val_dataset = OBSDataset(obs_directory, time_dir, lead_times_to_load, indices=val_indices)
      test_dataset = OBSDataset(obs_directory, time_dir, lead_times_to_load, indices=test_indices)
  
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
      for param in model.mlp_head.parameters():
          param.requires_grad = True
      model.cls_token.requires_grad = False
      batches_per_dataset = len(train_dataloader)
      print(f"batches per dataset: {batches_per_dataset}")
      optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
      
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.0001, factor=0.5, mode='min')
      #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
      
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
          for batch_data, batch_conditioning, batch_target in train_dataloader:
              batch_data = batch_data.to(device)
              batch_conditioning = batch_conditioning.to(device)
              batch_target = batch_target.to(device)
              y = model(batch_data, batch_conditioning)
              loss = loss_fn(y, batch_target)
              total_train_loss += loss.item() * batch_data.size(0)
          avg_train_loss = total_train_loss / len(train_dataloader.dataset)
          total_val_loss = 0.0
          for batch_data, batch_conditioning, batch_target in val_dataloader:
              batch_data = batch_data.to(device)
              batch_conditioning = batch_conditioning.to(device)
              batch_target = batch_target.to(device)
              pred_y = model(batch_data, batch_conditioning)
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
              #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
              """
              scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0 = 50,
                T_mult = 2,
                eta_min = 1e-7
              )
              """
              
          for batch_data, batch_conditioning, batch_target in train_dataloader:
              batch_data = batch_data.to(device)
              batch_conditioning = batch_conditioning.to(device)
              batch_target = batch_target.to(device)
              model.train()
              y = model(batch_data, batch_conditioning)
              loss = loss_fn(y, batch_target)
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              total_train_loss += loss.item() * batch_data.size(0)
          avg_train_loss = total_train_loss / len(train_dataloader.dataset)
          model.eval()
          total_val_loss = 0.0
          with torch.no_grad():
              for batch_data, batch_conditioning, batch_target in val_dataloader:
                  batch_data = batch_data.to(device)
                  batch_conditioning = batch_conditioning.to(device)
                  batch_target = batch_target.to(device)        
                  pred_y = model(batch_data, batch_conditioning)
                  loss = loss_fn(pred_y, batch_target)
                  total_val_loss += loss.item() * batch_data.size(0)
          avg_val_loss = total_val_loss / len(val_dataloader.dataset)
          if i >= lr_warmup_length:
              scheduler.step(avg_val_loss)
              #scheduler.step()
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
      torch.save(model, obs_out_dir + f"ViTTIMJO_FiLM_Andrew_OBS_leadTm{model_leadTms}_ensm{seed_num}_round{index + 1}.pth")
      compiled_train_losses.append(train_losses)
      compiled_val_losses.append(val_losses)
  
  print("\nCombined MDL and OBS Transfer Learning Training Completed.")
  
  import os
  import torch
  import numpy as np
  from netCDF4 import Dataset
  
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
      model_path = os.path.join(obs_out_dir, f"ViTTIMJO_FiLM_Andrew_OBS_leadTm{model_leadTms}_ensm{seed_num}_round{round_num}.pth")
      
      if not os.path.exists(model_path):
          print(f"WARNING: Model file not found, skipping round {round_num}. Path: {model_path}")
          continue
          
      print(f"  Loading model: {model_path}")
      model = torch.load(model_path, map_location=device, weights_only=False)
      model.to(device)
      
      # --- Get predictions for all datasets in this round using the helper function ---
      print("  Generating predictions for Train, Validation, and Test sets...")
      train_preds, train_lead_times, train_targets = get_all_preds_and_targets(model, train_dataloader, device)
      val_preds,   val_lead_times,   val_targets   = get_all_preds_and_targets(model, val_dataloader, device)
      test_preds,  test_lead_times,  test_targets  = get_all_preds_and_targets(model, test_dataloader, device)
      print("  Prediction generation complete.")
  
      # --- Write results to NetCDF files ---
      print("  Writing results to NetCDF files...")
      # The number of samples is now correctly retrieved from the length of the dataset in each dataloader
      # Save predictions and targets for each dataset type
      datasets_to_save = [
          ("Train", train_preds, train_targets, train_lead_times),
          ("Val",   val_preds,   val_targets,   val_lead_times),
          ("Test",  test_preds,  test_targets,  test_lead_times)
      ]
  
      for typ, preds, tgts, lead_times in datasets_to_save:
          save_to_netcdf(preds, tgts, lead_times, typ, bcor_directory, model_path)
  
  print("\nPrediction computation and NetCDF writing completed.")
  
  # --------------------
  # BCOR Calculation and Plotting Script for Multiple Lead Times
  # --------------------
  import os
  import csv
  import pandas as pd
  import matplotlib.pyplot as plt
  import torch
  import numpy as np
  from netCDF4 import Dataset
  
  # Define lead times and directories
  lead_times = lead_times_to_load
  bcor_directory = "Result_03/"
  
  num_ensembles = 1
  num_rounds = 3
  
  def torch_bcorr(x, y):
      pred_rmm1 = x[:, 0]
      pred_rmm2 = x[:, 1]
      target_rmm1 = y[:, 0]
      target_rmm2 = y[:, 1]
      return torch.sum(target_rmm1 * pred_rmm1 + target_rmm2 * pred_rmm2) / (
          torch.sqrt(torch.sum(target_rmm1 ** 2 + target_rmm2 ** 2)) * torch.sqrt(torch.sum(pred_rmm1 ** 2 + pred_rmm2 ** 2))
      )
  
  # Initialize results list
  results = []
  
  # Loop through each lead time
  for leadTm in lead_times:
      sum_train_preds = None
      sum_train_targets = None
      sum_test_preds = None
      sum_test_targets = None
      sum_val_preds = None
      sum_val_targets = None  
  
      for i in range(num_ensembles):
          train_preds = np.zeros((0, 2), dtype=np.float32)
          train_targets = np.zeros((0, 2), dtype=np.float32)
          test_preds = np.zeros((0, 2), dtype=np.float32)
          test_targets = np.zeros((0, 2), dtype=np.float32)
          val_preds = np.zeros((0, 2), dtype=np.float32)
          val_targets = np.zeros((0, 2), dtype=np.float32)
          
          for j in range(num_rounds):
              in1 = Dataset(bcor_directory + f"ViTTIMJO_FiLM_Andrew_OBS_leadTm{model_leadTms}_ensm{i+1}_round{j+1}_Prdct_Train_leadTm{leadTm}.nc")
              in2 = Dataset(bcor_directory + f"ViTTIMJO_FiLM_Andrew_OBS_leadTm{model_leadTms}_ensm{i+1}_round{j+1}_Prdct_Test_leadTm{leadTm}.nc")
              in3 = Dataset(bcor_directory + f"ViTTIMJO_FiLM_Andrew_OBS_leadTm{model_leadTms}_ensm{i+1}_round{j+1}_Prdct_Val_leadTm{leadTm}.nc")
              temp_train_pred = np.array(in1.variables["mjoTrainPred"][:])
              temp_train_target = np.array(in1.variables["mjoTrainTagt"][:])
              temp_test_pred = np.array(in2.variables["mjoTestPred"][:])
              temp_test_target = np.array(in2.variables["mjoTestTagt"][:])
              temp_val_pred = np.array(in3.variables["mjoValPred"][:])
              temp_val_target = np.array(in3.variables["mjoValTagt"][:])
              train_preds = np.vstack((train_preds, temp_train_pred))
              train_targets = np.vstack((train_targets, temp_train_target))
              test_preds = np.vstack((test_preds, temp_test_pred))
              test_targets = np.vstack((test_targets, temp_test_target))
              val_preds = np.vstack((val_preds, temp_val_pred))
              val_targets = np.vstack((val_targets, temp_val_target))
  
  
  
          if sum_train_preds is None:
              sum_train_preds = train_preds
              sum_train_targets = train_targets
              sum_test_preds = test_preds
              sum_test_targets = test_targets
              sum_val_preds = val_preds
              sum_val_targets = val_targets
          else:
              sum_train_preds += train_preds
              sum_train_targets += train_targets
              sum_test_preds += test_preds
              sum_test_targets += test_targets
              sum_val_preds += val_preds
              sum_val_targets += val_targets
  
      avg_train_preds = sum_train_preds / num_ensembles
      avg_train_targets = sum_train_targets / num_ensembles
      avg_test_preds = sum_test_preds / num_ensembles
      avg_test_targets = sum_test_targets / num_ensembles
      avg_val_preds = sum_val_preds / num_ensembles
      avg_val_targets = sum_val_targets / num_ensembles
  
      train_corr = torch_bcorr(torch.tensor(avg_train_preds), torch.tensor(avg_train_targets))
      test_corr = torch_bcorr(torch.tensor(avg_test_preds), torch.tensor(avg_test_targets))
      val_corr = torch_bcorr(torch.tensor(avg_val_preds), torch.tensor(avg_val_targets))
  
      print(f"Lead Time: {leadTm}, Train Correlation: {train_corr}, Test Correlation: {test_corr}, Val Correlation: {val_corr}")
  
      # Write BCOR results to CSV
      filename = bcor_directory + "FiLM_Time_BCOR_Multi(Long)_by_leadTm.csv"
      new_row = [leadTm, train_corr.item(), test_corr.item()]
      file_exists = os.path.isfile(filename)
  
      with open(filename, 'a', newline='') as csvfile:
          writer = csv.writer(csvfile)
          if not file_exists:
              writer.writerow(['leadTm', 'train_corr', 'test_corr'])
          writer.writerow(new_row)
  
      # Append results for plotting
      results.append(new_row)