import os

# Directories to check (relative to current working directory)
directories = [
    "Data/AllLeadTms",
    "Result_01/",
    "Result_02/",
    "Result_03/"
]

datadir = "/gpfs/gibbs/project/lu_lu/bec32/ML_for_MJO/Data/AllLeadTms/"

# Required files for MDL and OBS (leadTm=15 as in main script)
mdl_files = [
    datadir + f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_TMQ_leadTm15.nc",
    datadir + f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_FLUT_leadTm15.nc",
    datadir + f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_U200_leadTm15.nc",
    datadir + f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_U850_leadTm15.nc",
    datadir + f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_TREFHT_leadTm15.nc",
    datadir + f"CML2025_Step0C_MDL_remapped_90x180_daily_DJFM_nonFltr_PC1.nc",
    datadir + f"CML2025_Step0C_MDL_remapped_90x180_daily_DJFM_nonFltr_PC2.nc"
]
obs_files = [
    datadir + f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_tcwv_leadTm15.nc",
    datadir + f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_olr_leadTm15.nc",
    datadir + f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_u200_leadTm15.nc",
    datadir + f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_u850_leadTm15.nc",
    datadir + f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_trefht_leadTm15.nc",
    datadir + f"CML2025_Step0C_OBS_remapped_90x180_daily_DJFM_nonFltr_PC1.nc",
    datadir + f"CML2025_Step0C_OBS_remapped_90x180_daily_DJFM_nonFltr_PC2.nc"
]

# Required files for strong MJO predictions (Compute_Strong_Preds_Combined.py)
strong_pred_files = []
for round_num in [1, 2, 3]:
    strong_pred_files.extend([
        f"Result_03/ViTMJO_STRONG_CORR_Ben_OBS_leadTm15_ensm1_rund{round_num}_Prdct_Train.nc",
        f"Result_03/ViTMJO_STRONG_CORR_Ben_OBS_leadTm15_ensm1_rund{round_num}_Prdct_Val.nc",
        f"Result_03/ViTMJO_STRONG_CORR_Ben_OBS_leadTm15_ensm1_rund{round_num}_Prdct_Test.nc",
    ])

# Add Data/Time/ directory and LEADTM files
leadtm_dir = "Data/Time/"
leadtm_files = [
    os.path.join(leadtm_dir, "MDLleadTm15.nc"),
    os.path.join(leadtm_dir, "OBSleadTm15.nc")
]

# Add BCOR/STRONG_BCOR summary CSVs and standard prediction NetCDFs
summary_files = [
    "Result_03/BCOR_by_leadTm.csv",
    "Result_03/STRONG_BCOR_by_leadTm.csv"
]
prediction_files = []
for round_num in [1, 2, 3]:
    for split in ["Train", "Val", "Test"]:
        prediction_files.append(f"Result_03/ViTMJO_Ben_OBS_leadTm15_ensm1_rund{round_num}_Prdct_{split}.nc")

# Add Data/Time/ to directories
if leadtm_dir not in directories:
    directories.append(leadtm_dir)

print("Checking directories...")
for d in directories:
    if os.path.isdir(d):
        print(f"[OK] Directory exists: {d}")
    else:
        print(f"[MISSING] Directory does not exist: {d}")

print("\nChecking required MDL files...")
for f in mdl_files:
    if os.path.isfile(f):
        print(f"[OK] File exists: {f}")
    else:
        print(f"[MISSING] File does not exist: {f}")

print("\nChecking required OBS files...")
for f in obs_files:
    if os.path.isfile(f):
        print(f"[OK] File exists: {f}")
    else:
        print(f"[MISSING] File does not exist: {f}")

print("\nChecking strong MJO prediction files (Compute_Strong_Preds_Combined.py outputs)...")
for f in strong_pred_files:
    if os.path.isfile(f):
        print(f"[OK] File exists: {f}")
    else:
        print(f"[MISSING] File does not exist: {f}")

print("\nChecking Data/Time/ directory and LEADTM files...")
if os.path.isdir(leadtm_dir):
    print(f"[OK] Directory exists: {leadtm_dir}")
else:
    print(f"[MISSING] Directory does not exist: {leadtm_dir}")
for f in leadtm_files:
    if os.path.isfile(f):
        print(f"[OK] File exists: {f}")
    else:
        print(f"[MISSING] File does not exist: {f}")

print("\nChecking BCOR/STRONG_BCOR summary CSVs...")
for f in summary_files:
    if os.path.isfile(f):
        print(f"[OK] File exists: {f}")
    else:
        print(f"[MISSING] File does not exist: {f}")

print("\nChecking standard prediction NetCDF files...")
for f in prediction_files:
    if os.path.isfile(f):
        print(f"[OK] File exists: {f}")
    else:
        print(f"[MISSING] File does not exist: {f}")

# Add summary at the end
def all_ok():
    missing = False
    # Check all directories
    for d in directories:
        if not os.path.isdir(d):
            missing = True
    # Check all files
    for filelist in [mdl_files, obs_files, strong_pred_files, leadtm_files, summary_files, prediction_files]:
        for f in filelist:
            if not os.path.isfile(f):
                missing = True
    return not missing

if all_ok():
    print("\nAll required files and directories are present!")
else:
    print("\nSome files or directories are missing. Please check above.")

print("\nCheck complete.")
