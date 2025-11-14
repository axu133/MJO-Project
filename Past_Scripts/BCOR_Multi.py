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
lead_times = [10,15,20,25,30]  # Example lead times
leadTm_name = "10-30"
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
            in1 = Dataset(bcor_directory + f"ViTTIMJO_Andrew_OBS_leadTm{leadTm_name}_ensm{i+1}_round{j+1}_Prdct_Train_leadTm{leadTm}.nc")
            in2 = Dataset(bcor_directory + f"ViTTIMJO_Andrew_OBS_leadTm{leadTm_name}_ensm{i+1}_round{j+1}_Prdct_Test_leadTm{leadTm}.nc")
            in3 = Dataset(bcor_directory + f"ViTTIMJO_Andrew_OBS_leadTm{leadTm_name}_ensm{i+1}_round{j+1}_Prdct_Val_leadTm{leadTm}.nc")
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
    filename = bcor_directory + "Time_BCOR_Multi_by_leadTm.csv"
    new_row = [leadTm, train_corr.item(), test_corr.item()]
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['leadTm', 'train_corr', 'test_corr'])
        writer.writerow(new_row)

    # Append results for plotting
    results.append(new_row)