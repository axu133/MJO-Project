import netCDF4
import os
import numpy as np

def display_netcdf_properties(nc_file_path):
    with netCDF4.Dataset(nc_file_path, 'r') as nc:
        print("File format:", nc.file_format)
        print("Dimensions:")
        for dim in nc.dimensions.values():
            print(f"  {dim.name}: {len(dim)}")
        print("\nVariables:")
        for var in nc.variables.values():
            print(f"  {var.name}: {var.dtype}, dimensions: {var.dimensions}, shape: {var.shape}")
        print("\nGlobal Attributes:")
        for attr in nc.ncattrs():
            print(f"  {attr}: {getattr(nc, attr)}")

def display_netcdf_mean_std(nc_file_path):
    """
    Display the mean and standard deviation of all numeric variables in a NetCDF file.
    """
    import numpy as np
    with netCDF4.Dataset(nc_file_path, 'r') as nc:
        print(f"\nMean and Std for variables in: {nc_file_path}")
        for var_name, var in nc.variables.items():
            # Only process numeric variables (skip char/str, etc.)
            if hasattr(var, 'dtype') and np.issubdtype(var.dtype, np.number):
                try:
                    data = var[:]
                    mean = np.nanmean(data)
                    std = np.nanstd(data)
                    print(f"  {var_name}: mean={mean:.4f}, std={std:.4f}")
                except Exception as e:
                    print(f"  {var_name}: Could not compute (error: {e})")
            else:
                print(f"  {var_name}: (non-numeric, skipped)")

def create_constant_netcdf(input_nc_path, leadTm, output_filename):
    # Prepare output directory and filename
    output_dir = os.path.join('Data', 'Time')
    os.makedirs(output_dir, exist_ok=True)
    output_nc_path = os.path.join(output_dir, output_filename)
    leadtm_val = leadTm / 40.0
    with netCDF4.Dataset(input_nc_path, 'r') as src:
        with netCDF4.Dataset(output_nc_path, 'w') as dst:
            # Copy only time, lat, lon dimensions
            for name, dim in src.dimensions.items():
                dst.createDimension(name, (len(dim) if not dim.isunlimited() else None))
            # Copy only time, lat, lon variables (not data variables)
            for name, var in src.variables.items():
                if name in ['time', 'lat', 'lon']:
                    out_var = dst.createVariable(name, var.datatype, var.dimensions)
                    # Set attributes before writing data
                    """
                    out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                    out_var[:] = var[:]
                    """
                    
                    
                    # Set attributes before writing data
                    out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                    
                    # Debug: print shapes and dtypes before assignment
                    #print(f"Copying variable: {name}")
                    #print("  src shape:", var.shape, "dst shape:", out_var.shape)
                    #print("  src dims:", var.dimensions, "dst dims:", out_var.dimensions)
                    #print("  src dtype:", var.dtype, "dst dtype:", out_var.dtype)
                    
                    # Ensure shape and dtype match
                    data_to_write = var[:]
                    if data_to_write.shape != out_var.shape:
                        try:
                            data_to_write = data_to_write.reshape(out_var.shape)
                        except Exception as e:
                            print(f"Shape mismatch for variable {name}: cannot reshape {data_to_write.shape} to {out_var.shape}")
                            raise
                    if data_to_write.dtype != out_var.dtype:
                        data_to_write = data_to_write.astype(out_var.dtype)
                    out_var[:] = data_to_write
                    
                    
            # Add new LEADTM variable with shape (time, lat, lon)
            time_len = len(dst.dimensions['time'])
            lat_len = len(dst.dimensions['lat'])
            lon_len = len(dst.dimensions['lon'])
            leadtm_var = dst.createVariable('LEADTM', 'f4', ('time', 'lat', 'lon'))
            leadtm_var[:, :, :] = leadtm_val
            leadtm_var.units = f"leadTm/40.0 (input={leadTm})"
            # Copy global attributes
            dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
    #print(f"Saved: {output_nc_path}")

# Example usage:
# input_nc = "D:/Research/MJO-ML/CML2025_Step0C_MDL_remapped_90x180_daily_DJFM_nonFltr_PC1_leadTm34.nc"
# lead_time = 15
# create_constant_netcdf(input_nc, lead_time)

datadir = "/gpfs/gibbs/project/lu_lu/bec32/ML_for_MJO/Data/AllLeadTms/"
for leadTm in range(1,41):
    #create_constant_netcdf(datadir + f"CML2025_Step0C_TROP30_MDL_remapped_90x180_daily_DJFM_Anom_nonFltr_U200_leadTm{leadTm}.nc", leadTm, f"MDLleadTm{leadTm}.nc")
    #create_constant_netcdf(datadir + f"CML2025_Step0C_TROP30_OBS_remapped_90x180_daily_DJFM_Anom_nonFltr_u200_leadTm{leadTm}.nc", leadTm, f"OBSleadTm{leadTm}.nc")
    #print(f"Files created for lead time {leadTm}")
    display_netcdf_mean_std(f"Data/Time/MDLleadTm{leadTm}.nc")
    display_netcdf_mean_std(f"Data/Time/OBSleadTm{leadTm}.nc")
