# --------------------
# BCOR Calculation and Plotting Script (No Model Training)
# --------------------
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from netCDF4 import Dataset

# Set lead time and directories (edit as needed)
leadTm = 15  # Set to match the lead time used in predictions
bcor_directory = "Result_03/"

num_ensembles = 1
num_rounds = 3
# The following numbers should match your prediction/test set sizes
sum_train_preds = None
sum_train_targets = None
sum_test_preds = None
sum_test_targets = None

def torch_bcorr(x, y):
    pred_rmm1 = x[:,0]
    pred_rmm2 = x[:,1]
    target_rmm1 = y[:,0]
    target_rmm2 = y[:,1]
    return torch.sum(target_rmm1 * pred_rmm1 + target_rmm2 * pred_rmm2) / (torch.sqrt(torch.sum(target_rmm1 ** 2 + target_rmm2**2)) * torch.sqrt(torch.sum(pred_rmm1 ** 2 + pred_rmm2**2)))

for i in range(num_ensembles):
    train_preds = np.zeros((0, 2), dtype=np.float32)
    train_targets = np.zeros((0, 2), dtype=np.float32)
    test_preds = np.zeros((0, 2), dtype=np.float32)
    test_targets = np.zeros((0, 2), dtype=np.float32)
    for j in range(num_rounds):
        in1 = Dataset(bcor_directory + f"ViTTIMJO_Andrew_OBS_leadTm{leadTm}_ensm{i+1}_rund{j+1}_Prdct_Train.nc")
        in2 = Dataset(bcor_directory + f"ViTTIMJO_Andrew_OBS_leadTm{leadTm}_ensm{i+1}_rund{j+1}_Prdct_Test.nc")
        temp_train_pred = np.array(in1.variables["mjoTrainPred"][:])
        temp_train_target = np.array(in1.variables["mjoTrainTagt"][:])
        temp_test_pred = np.array(in2.variables["mjoTestPred"][:])
        temp_test_target = np.array(in2.variables["mjoTestTagt"][:])
        train_preds = np.vstack((train_preds, temp_train_pred))
        train_targets = np.vstack((train_targets, temp_train_target))
        test_preds = np.vstack((test_preds, temp_test_pred))
        test_targets = np.vstack((test_targets, temp_test_target))
    # Dynamically initialize sum arrays based on shape
    if sum_train_preds is None:
        sum_train_preds = np.zeros_like(train_preds)
        sum_train_targets = np.zeros_like(train_targets)
        sum_test_preds = np.zeros_like(test_preds)
        sum_test_targets = np.zeros_like(test_targets)
    sum_train_preds = sum_train_preds + train_preds
    sum_train_targets = sum_train_targets + train_targets
    sum_test_preds = sum_test_preds + test_preds
    sum_test_targets = sum_test_targets + test_targets

avg_train_preds = sum_train_preds / num_ensembles
avg_train_targets = sum_train_targets / num_ensembles
avg_test_preds = sum_test_preds / num_ensembles
avg_test_targets = sum_test_targets / num_ensembles

train_corr = torch_bcorr(torch.tensor(avg_train_preds), torch.tensor(avg_train_targets))
print(f"Train Correlation: {train_corr}")

test_corr = torch_bcorr(torch.tensor(avg_test_preds), torch.tensor(avg_test_targets))
print(f"Test Correlation: {test_corr}")

# Write BCOR results to CSV
filename = bcor_directory + "Time_BCOR_by_leadTm.csv"
new_row = [leadTm, train_corr.item(), test_corr.item()]
file_exists = os.path.isfile(filename)

with open(filename, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(['leadTm', 'train_corr', 'test_corr'])
    writer.writerow(new_row)

print("\nBivariate Correlation Testing Completed.")

# Plot BCOR Results
bcor_csv = bcor_directory + "Time_BCOR_by_leadTm.csv"
if os.path.isfile(bcor_csv):
    df = pd.read_csv(bcor_csv)
    plt.figure(figsize=(8,5))
    plt.plot(df['leadTm'], df['train_corr'], marker='o', label='Train BCOR')
    plt.plot(df['leadTm'], df['test_corr'], marker='s', label='Test BCOR')
    plt.axhline(0.5, color='red', linestyle='--', label='BCOR = 0.5')
    plt.xlabel('Lead Time')
    plt.ylabel('Bivariate Correlation (BCOR)')
    plt.title('Bivariate Correlation by Lead Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print(f"BCOR results file not found: {bcor_csv}")