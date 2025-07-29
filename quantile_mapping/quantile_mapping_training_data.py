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
from src.utils_essential import *

from ibicus.debias import QuantileMapping, QuantileDeltaMapping, ISIMIP

from src.dataloader_sr import ERA5_P_0_25_to_1_Dataset, GFDL_P_Dataset_64_1992_2014, QM_GFDL_LR_Dataset_64
# -

# # Prepare dataloaders (QM of 1d data)

bs_valid = 1400

# +
gfdl_train = GFDL_P_Dataset_64_1992_2014(stage='train')

dataloader_gfdl_train = data.DataLoader(gfdl_train.data(), batch_size=6936, shuffle=False,
                                        drop_last=True,num_workers=2)

gfdl_lr_64_train_for_qm = next(iter(dataloader_gfdl_train)).unsqueeze(dim=1)
print("GFDL LR 64 shape:",gfdl_lr_64_train_for_qm.shape)
# -

era5_p_0_25_to_1_t = ERA5_P_0_25_to_1_Dataset(stage='train')
dataloader_era5_train_1d_64 = data.DataLoader(era5_p_0_25_to_1_t.data(), batch_size=6936
                                              , shuffle=False, drop_last=True,num_workers=2)
era5_train = next(iter(dataloader_era5_train_1d_64)).unsqueeze(dim=1)
era5_train.shape

# +
gfdl = GFDL_P_Dataset_64_1992_2014(stage='valid')

dataloader_gfdl_valid = data.DataLoader(gfdl.data(), batch_size=bs_valid, shuffle=False,
                                  drop_last=True,num_workers=2)


gfdl_lr_64_valid_for_qm = next(iter(dataloader_gfdl_valid)).unsqueeze(dim=1)
print("GFDL LR 64 shape:",gfdl_lr_64_valid_for_qm.shape)

# +
era5_p_0_25_to_1_v = ERA5_P_0_25_to_1_Dataset(stage='valid')

dataloader_era5_val_1d_64 = data.DataLoader(era5_p_0_25_to_1_v.data(), batch_size=bs_valid,
                                            shuffle=False, drop_last=True,num_workers=2)

era5_lr_64 = next(iter(dataloader_era5_val_1d_64)).unsqueeze(1)
print("ERA5 LR 64 shape:",era5_lr_64.shape)

# +
from src.dataloader_sr import original_era5

original_era5_ds = original_era5("train")
dl_original_era5 = data.DataLoader(original_era5_ds.data(), batch_size=6936, shuffle=False,
                                  drop_last=True,num_workers=2)
era5_orginal = next(iter(dl_original_era5)).unsqueeze(dim=1).numpy()
# -

# # Apply QM debiasing to gfdl validation 

# ### apply QM

train_era5 = era5_p_0_25_to_1_t.inverse_dwd_trafo(era5_train)[:6936,:,:,:].squeeze(1).numpy()
train_gfdl = gfdl_train.inverse_dwd_trafo(gfdl_lr_64_train_for_qm)[:6936,:,:,:].squeeze(1).numpy()
valid_era5 = era5_p_0_25_to_1_v.inverse_dwd_trafo(era5_lr_64)[:1400,:,:,:].squeeze(1).numpy() 
valid_gfdl = gfdl.inverse_dwd_trafo(gfdl_lr_64_valid_for_qm)[:1400,:,:,:].squeeze(1).numpy()

# +
debiased_valid_gfdl = np.zeros((1400, 64, 64)) 

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

latitudinal_mean_three_np( debiased_valid_gfdl[:,np.newaxis,:,:]
                            ,valid_gfdl[:,np.newaxis,:,:]
                            ,valid_era5[:,np.newaxis,:,:]
                            ,label_name = [ "debiased train gfdl ","train gfdl", "train era5"])

histograms_three_np(debiased_valid_gfdl[:500,np.newaxis,:,:]
                ,valid_gfdl[:500,np.newaxis,:,:]
                ,valid_era5[:500,np.newaxis,:,:]
                ,label_name = ["debiased valid gfdl ", "valid gfdl", "valid era5"]
                ,xlim_end=150,bins=300)

# +
ssd = SpatialSpectralDensity_diff_res(   debiased_valid_gfdl[:,np.newaxis,:,:]
                                        ,valid_gfdl[:,np.newaxis,:,:]
                                        ,valid_era5[:,np.newaxis,:,:]
                                        ,new_labels = [ "debiased valid gfdl ",
                                                       "valid gfdl", "valid era5"])
ssd.run(num_times=None)

ssd.plot_psd(fname='',model_resolution=1,model_resolution_2=1)
# -

latitudinal_mean_three_np(   valid_era5[:,np.newaxis,:,:]
                            ,debiased_valid_gfdl[:,np.newaxis,:,:]
                            ,era5_orginal
                            ,label_name = [ "loaded debiased tr gfdl ",
                                       "debiased tr gfdl", "original era5"])

print("debiased gfdl valid")
plot_images_no_lab(torch.tensor(debiased_valid_gfdl).unsqueeze(1)[:5])
print("gfdl valid")
plot_images_no_lab(torch.tensor(valid_gfdl).unsqueeze(1)[:5])

# # QM on train GFDL

# +
do_qm_on_train_set = False

if do_qm_on_train_set == True:
    debiased_train_gfdl = np.zeros((6936, 64, 64)) 

    # Create synthetic time coordinates
    time_coords_long = xr.cftime_range("2000-01-01", periods=6936, freq="D", calendar="gregorian")


    for lat in range(train_era5.shape[1]):
        for lon in range(train_era5.shape[2]):

            qm = sdba.adjustment.QuantileDeltaMapping

            ref_data = train_era5[:, lat, lon]
            hist_data = train_gfdl[:, lat, lon]
            input_data = train_gfdl[:, lat, lon]

            # Pass DataArray instead of Dataset
            ref_data_array = xr.DataArray(ref_data, dims="time", coords={"time": time_coords_long}, attrs={"units": "dimensionless"})
            hist_data_array = xr.DataArray(hist_data, dims="time", coords={"time": time_coords_long}, attrs={"units": "dimensionless"})
            input_data_array = xr.DataArray(input_data, dims="time", coords={"time": time_coords_long}, attrs={"units": "dimensionless"})

            # Specify the group and other parameters
            group = sdba.adjustment.Grouper("time")

            Adj = qm.train(
                ref_data_array,
                hist=hist_data_array,
                nquantiles=500,
                group=group,
                skip_input_checks=True
            )


            mapped = Adj.adjust(input_data_array, skip_input_checks=False)
            debiased_train_gfdl[:, lat, lon] = mapped

    print("Processing completed.")
# -

if do_qm_on_train_set == True:
    histograms_three_np(debiased_train_gfdl[:500,np.newaxis,:,:]
                        ,train_gfdl[:500,np.newaxis,:,:]
                        ,train_era5[:500,np.newaxis,:,:]
                        ,label_name = ["debiased train gfdl ", "train gfdl", "train era5"]
                        ,xlim_end=150,bins=300)

# ## save QM gfdl valid set

np.min(debiased_train_gfdl), np.max(debiased_train_gfdl)

do_clipping_before_saving = True
if do_clipping_before_saving == True:
    debiased_train_gfdl = np.clip(debiased_train_gfdl,0,700)
    debiased_valid_gfdl = np.clip(debiased_valid_gfdl,0,700)

# +
do_save_qm_gfdl = False

if do_save_qm_gfdl == True: 
    ### comment in when really wanting to save
    torch.save(torch.tensor(debiased_train_gfdl).to("cuda"), 'data/11_01_deltaQM_debiased_gfdl_train_custom_dl.pth')

    torch.save(torch.tensor(debiased_valid_gfdl).to("cuda"), 'data/11_01_deltaQM_debiased_gfdl_valid_custom_dl.pth')
    print("saving quantile mapped gfdl traing dataset")
# -

# ## load QM data

# +
gfdl_valid_qm = QM_GFDL_LR_Dataset_64('data/11_01_deltaQM_debiased_gfdl_valid_custom_dl.pth')

dataloader_gfdl_valid_qm = data.DataLoader(gfdl_valid_qm.data(), batch_size=bs_valid, shuffle=False,
                                  drop_last=True,num_workers=2)


gfdl_lr_64_valid_qm = next(iter(dataloader_gfdl_valid_qm)).unsqueeze(dim=1)
print("GFDL LR 64 shape:",gfdl_lr_64_valid_qm.shape)
torch.min(gfdl_lr_64_valid_qm), torch.max(gfdl_lr_64_valid_qm)
# -

latitudinal_mean_three_np( gfdl_valid_qm.inverse_dwd_trafo(gfdl_lr_64_valid_qm.numpy())
                            ,debiased_valid_gfdl[:,np.newaxis,:,:]
                            ,valid_era5[:,np.newaxis,:,:]
                            ,label_name = [ "loaded debiased valid gfdl ",
                                       "debiased valid gfdl", "valid era5"])

latitudinal_mean_three_np( gfdl_lr_64_valid_qm.numpy()
                            ,era5_lr_64.numpy()
                            ,era5_lr_64.numpy()
                            ,label_name = [ "qm valid gfdl ", "train era5", "train era5"])

# +
from src.dataloader_custom import original_era5

original_era5_ds = original_era5("train")
dl_original_era5 = data.DataLoader(original_era5_ds.data(), batch_size=6936, shuffle=False,
                                  drop_last=True,num_workers=2)
era5_orginal = next(iter(dl_original_era5)).unsqueeze(dim=1).numpy()

# +
eval_training_ds_qm = False

if eval_training_ds_qm == True:
    gfdl_tr_qm = QM_GFDL_LR_Dataset_64('data/11_01_deltaQM_debiased_gfdl_valid_custom_dl.pth')

    dataloader_gfdl_tr_qm = data.DataLoader(gfdl_tr_qm.data(), batch_size=6936, shuffle=False,
                                      drop_last=True,num_workers=2)


    gfdl_lr_64_tr_qm = next(iter(dataloader_gfdl_tr_qm)).unsqueeze(dim=1)
    print("GFDL tr LR 64 shape:",gfdl_lr_64_tr_qm.shape)
    torch.min(gfdl_lr_64_tr_qm), torch.max(gfdl_lr_64_tr_qm)
# -

if eval_training_ds_qm == True:
    latitudinal_mean_three_np(   gfdl_tr_qm.inverse_dwd_trafo(gfdl_lr_64_tr_qm.numpy())
                                ,debiased_train_gfdl[:,np.newaxis,:,:]
                                ,era5_orginal
                                ,label_name = [ "loaded debiased tr gfdl ",
                                           "debiased tr gfdl", "original era5"])
    latitudinal_mean_three_np(   gfdl_tr_qm.inverse_dwd_trafo(gfdl_lr_64_tr_qm.numpy())
                                ,debiased_train_gfdl[:,np.newaxis,:,:]
                                ,era5_p_0_25_to_1_t.inverse_dwd_trafo(era5_train).numpy()-1
                                ,label_name = [ "loaded debiased tr gfdl ",
                                           "debiased tr gfdl", "tr era5"])
    
    latitudinal_mean_three_np(   gfdl_lr_64_tr_qm.numpy()
                            ,gfdl_lr_64_train_for_qm.numpy()
                            ,era5_train.numpy()
                            ,label_name = [ "loaded debiased tr gfdl", "raw tr gfdl", "tr era5"])
    
    print("valid gfdl")
    plot_images_no_lab(gfdl_lr_64_valid_for_qm[:5])
    print("debiased gfdl valid")
    plot_images_no_lab(gfdl_lr_64_valid_qm[:5])


