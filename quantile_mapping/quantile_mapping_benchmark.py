# # SETUP

# %%capture
# !pip install xarray
# !pip install wandb
# !pip install netcdf4
# !pip install collections
# !pip install scikit-image
# !pip install pysteps
# !pip install xclim
# !pip install ibicus

# %env CUDA_VISIBLE_DEVICES=0,1

# +
import numpy as np
from torch.utils import data
from src.utils import *
import xarray as xr
from scipy.ndimage import convolve
from skimage.transform import rescale
import torch
from xclim import sdba

from src.utils import *
from src.utils_essential import *
from ibicus.debias import QuantileMapping, QuantileDeltaMapping, ISIMIP

from src.dataloader_sr import gfdl_eval_256, era5_upscaled_1d_256, era5_0_25d_256, gfdl_eval_256_ssp585
# -

# # Prepare dataloaders (256 = 0.25d)

bs_train = 6936
bs_valid = 1400

# +
era5_hr_ds = era5_0_25d_256(stage='train')
era5_hr_dl = data.DataLoader(era5_hr_ds, batch_size=bs_train, shuffle=False, drop_last=True)

era5_hr = next(iter(era5_hr_dl))
print(era5_hr.shape)

# +
gfdl_256_ds = gfdl_eval_256(stage='train')
gfdl_256_dl = data.DataLoader(gfdl_256_ds, batch_size=bs_train, shuffle=False, drop_last=True)

gfdl_256 = next(iter(gfdl_256_dl))
print(gfdl_256.shape)

# +
gfdl_256_ds_v = gfdl_eval_256(stage='valid')
gfdl_256_dl_v = data.DataLoader(gfdl_256_ds_v, batch_size=bs_valid, shuffle=False, drop_last=True)

gfdl_256_v = next(iter(gfdl_256_dl_v))
print(gfdl_256_v.shape)

# +
era5_hr_ds_v = era5_0_25d_256(stage='valid')
era5_hr_dl_v = data.DataLoader(era5_hr_ds_v, batch_size=bs_valid, shuffle=False, drop_last=True)

era5_hr_v = next(iter(era5_hr_dl_v))
print(era5_hr_v.shape)

# +
era5_lr_ds = era5_upscaled_1d_256(stage='valid')
era5_lr_dl = data.DataLoader(era5_lr_ds, batch_size=bs_valid, shuffle=False, drop_last=True)

era5_lr = next(iter(era5_lr_dl))
print(era5_lr.shape)
# -

# # Apply QM debiasing to gfdl validation (QM on 0.25d data)

# ### apply QM

train_era5 = era5_hr_ds.inverse_dwd_trafo(era5_hr)[:6936,:,:,:].squeeze(1).numpy()
train_gfdl = gfdl_256_ds.inverse_dwd_trafo(gfdl_256)[:6936,:,:,:].squeeze(1).numpy()
valid_era5 = era5_hr_ds_v.inverse_dwd_trafo(era5_hr_v)[:1400,:,:,:].squeeze(1).numpy() 
valid_gfdl = gfdl_256_ds_v.inverse_dwd_trafo(gfdl_256_v)[:1400,:,:,:].squeeze(1).numpy()

train_era5.shape, valid_era5.shape

# +
debiased_valid_gfdl = np.zeros((1400, 256, 256)) 

# Create synthetic time coordinates
time_coords_long = xr.cftime_range("1992-01-01", periods=6936, freq="D", calendar="gregorian")
time_coords_short = xr.cftime_range("2011-01-01", periods=1400, freq="D", calendar="gregorian")


for lat in range(train_era5.shape[1]):
    for lon in range(train_era5.shape[2]):

        qm = sdba.adjustment.QuantileDeltaMapping
        #qm = sdba.adjustment.EmpiricalQuantileMapping
        

        ref_data = train_era5[:, lat, lon]
        hist_data = train_gfdl[:, lat, lon]
        input_data = valid_gfdl[:, lat, lon]

        # Pass DataArray instead of Dataset
        ref_data_array = xr.DataArray(ref_data, dims="time", coords={"time": time_coords_long}, attrs={"units": "dimensionless"})
        hist_data_array = xr.DataArray(hist_data, dims="time", coords={"time": time_coords_long}, attrs={"units": "dimensionless"})
        input_data_array = xr.DataArray(input_data, dims="time", coords={"time": time_coords_short}, attrs={"units": "dimensionless"})

        # Specify the group and other parameters
        group = sdba.adjustment.Grouper("time")

        Adj = qm.train(
            ref_data_array,
            hist=hist_data_array,
            nquantiles=100,      # 100 was good
            group=group,
            skip_input_checks=True
        )


        mapped = Adj.adjust(input_data_array, skip_input_checks=False)
        debiased_valid_gfdl[:, lat, lon] = mapped

print("Processing completed.")
# -

np.min(debiased_valid_gfdl), np.max(debiased_valid_gfdl) 

# +
values = debiased_valid_gfdl.flatten()

plt.hist(values, bins=500, color='blue', edgecolor='black')

plt.title('Histogram of debiased_valid_gfdl Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.xlim(left=-20, right=40)

plt.show()
# -

count_negative_entries = np.sum(debiased_valid_gfdl < 0)
print(f"Number of entries smaller than 0: {count_negative_entries}")

debiased_valid_gfdl.shape

latitudinal_mean_three( torch.tensor(debiased_valid_gfdl[:,np.newaxis,:,:])
                        ,torch.tensor(valid_gfdl[:,np.newaxis,:,:])
                        ,torch.tensor(valid_era5[:,np.newaxis,:,:])
                        ,label_name = [ "debiased train gfdl ","train gfdl", "train era5"])

histograms_three_np(debiased_valid_gfdl[:500,np.newaxis,:,:]
                    ,valid_gfdl[:500,np.newaxis,:,:]
                    ,valid_era5[:500,np.newaxis,:,:]
                    ,label_name = ["debiased valid gfdl ", "valid gfdl", "valid era5"]
                    ,xlim_end=200,bins=300)

# +
ssd = SpatialSpectralDensity_diff_res(   debiased_valid_gfdl[:,np.newaxis,:,:]
                                        ,valid_gfdl[:,np.newaxis,:,:]
                                        ,valid_era5[:,np.newaxis,:,:]
                                        ,new_labels = [ "debiased valid gfdl ",
                                                       "valid gfdl", "valid era5"])
ssd.run(num_times=None)

ssd.plot_psd(fname='',model_resolution=1,model_resolution_2=1)
# -

print("debiased gfdl valid")
plot_images_no_lab(torch.tensor(debiased_valid_gfdl).unsqueeze(1)[:5])
print("gfdl valid")
plot_images_no_lab(torch.tensor(valid_gfdl).unsqueeze(1)[:5])

# ## saving

do_clipping_before_saving = True
if do_clipping_before_saving == True:
    debiased_valid_gfdl = np.clip(debiased_valid_gfdl,0,700)

# +
do_save_qm_gfdl = False

if do_save_qm_gfdl == True: 

    torch.save(torch.tensor(debiased_valid_gfdl).to("cuda"), 'data/QM_hr_debiased_gfdl_valid_clip_0.pth')
    print("saving quantile mapped gfdl traing dataset")
# -

debiased_valid_gfdl.shape
