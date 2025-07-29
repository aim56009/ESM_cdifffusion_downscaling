# %%capture
# !pip install xarray
# !pip install scikit-image
# !pip install netcdf4

import xarray as xr
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from skimage.transform import rescale
from scipy.ndimage import convolve
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

import os
new_directory = '/dss/dsshome1/0D/ge74xuf2/climate_diffusion'
os.chdir(new_directory)
os.getcwd()


# # precipitation

def dwd_rv_rainrate_transform_no_clip(x):
    #x = raw.copy()
    x += 1                              
    x = np.log10(x, out=x)                        
    mu = np.mean(x)
    std = np.std(x)
    x -= mu       
    x /= std
    return x, mu, std


# # QM GFDL - use HR ERA5 mean,std to trafo data

mean_hr_era5       = 0.3700341
std_hr_era5        = 0.39072984
min_value_hr_era5  = -0.9470331
max_value_hr_era5  = 6.000293


def dwd_rv_rainrate_transform_no_clip(x):
    x += 1                              
    x = np.log10(x, out=x)                        
    
    x -= mean_hr_era5       
    x /= std_hr_era5
    return x, mean_hr_era5, std_hr_era5


def transform_bc_gfdl_to_era5_units(do_save=False):
    path = 'data/QM_hr_debiased_gfdl_valid_clip_0.pth'
    data_ = torch.load(path).cpu().numpy()                                      # load data
    data_, mean, std = dwd_rv_rainrate_transform_no_clip(data_)                 # log(x+1) -> standardize 
    data_min = min_value_hr_era5
    data_max = max_value_hr_era5
    data_ = (data_ - data_.min()) / (data_.max() - data_.min())                 # bring data to [-1,1]
    data_ = data_ * 2 - 1 

    data_ = torch.tensor(data_).float()
    print("data.shape:",data_.shape)
    
    if do_save==True:
        print("saving")
        torch.save(data_,"/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/qm_gfdl_256_era5_units.nc")
    
    return data_, mean, std, data_min, data_max


# +
do_qm_gfdl_era5_trafo = True

if do_qm_gfdl_era5_trafo == True:
    data_rescaled, mean, std, minval, maxval = transform_bc_gfdl_to_era5_units(do_save=True)
    print(mean, std, minval, maxval)


# -

class qm_gfdl_era5_units_hr(data.Dataset):
    def __init__(self):
    
        self.era5_path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/qm_gfdl_256_era5_units.nc"
            
        self.era5 = None


    def __getitem__(self, index):
        if self.era5 is None:
            self.era5 = torch.load(self.era5_path)
        return torch.unsqueeze(self.era5[index, :, :],0)

    def __len__(self):
        if self.era5 is None:
            self.era5 = torch.load(self.era5_path)
        return self.era5[:,0,0].shape[0]
    
    def inverse_dwd_trafo(self, transformed_data):
        mean = 0.3700341
        std = 0.39072984
        min_value = -0.9470331
        max_value = 6.000293

        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        x *= std
        x += mean
        x = 10 ** x
        x -=1
        return x

# +
era5_new_ds_ = qm_gfdl_era5_units_hr()

era5_new_dl_ = data.DataLoader(era5_new_ds_, batch_size=10, shuffle=False, drop_last=True)

era5_new_sample_ = next(iter(era5_new_dl_))
print(era5_new_sample_.shape)
# -

qm_gfdl_original = torch.load('data/QM_hr_debiased_gfdl_valid_clip_0.pth').unsqueeze(1)[:10]
qm_gfdl_original.shape

from src.utils import *
latitudinal_mean_three(qm_gfdl_original
                      ,qm_gfdl_original
                      ,era5_new_ds_.inverse_dwd_trafo(era5_new_sample_) 
                      ,label_name=["ori","ori", "trafo-units-back"])

plt.imshow(era5_new_ds_.inverse_dwd_trafo(era5_new_sample_.numpy()) [0,0,:,:])


# # QM GFDL (use datas min,max,mean,std -not era5)

def transform_bc_gfdl_to_trafo_units(do_save=False):
    path = 'data/QM_hr_debiased_gfdl_valid_clip_0.pth'
    data_ = torch.load(path).cpu().numpy()                                      # load data
    data_, mean, std = dwd_rv_rainrate_transform_no_clip(data_)                 # log(x+1) -> standardize 
    data_min = data_.min()
    data_max = data_.max()
    data_ = (data_ - data_.min()) / (data_.max() - data_.min())                 # bring data to [-1,1]
    data_ = data_ * 2 - 1 

    data_ = torch.tensor(data_).float()
    print("data.shape:",data_.shape)
    
    if do_save==True:
        print("saving")
        torch.save(data_,"/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/qm_gfdl_256_trafo_units.nc")
    
    return data_, mean, std, data_min, data_max


# +
do_test_functions_gfdl_64 = True

if do_test_functions_gfdl_64 == True:
    data_rescaled, mean, std, minval, maxval = transform_bc_gfdl_to_trafo_units(do_save=False)
    print(mean, std, minval, maxval)


# -

class qm_gfdl_trafo_units_hr(data.Dataset):
    def __init__(self):
    
        self.era5_path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/qm_gfdl_256_trafo_units.nc"
            
        self.era5 = None


    def __getitem__(self, index):
        if self.era5 is None:
            self.era5 = torch.load(self.era5_path)
        return torch.unsqueeze(self.era5[index, :, :],0)

    def __len__(self):
        if self.era5 is None:
            self.era5 = torch.load(self.era5_path)
        return self.era5[:,0,0].shape[0]
    
    def inverse_dwd_trafo(self, transformed_data):
        mean=0.36671052005892085
        std=0.3896533600327316
        min_value=-0.941119871334143
        max_value= 5.396119777081304
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        x *= std
        x += mean
        x = 10 ** x
        x -=1
        return x

# +
era5_new_ds_ = qm_gfdl_trafo_units_hr()

era5_new_dl_ = data.DataLoader(era5_new_ds_, batch_size=10, shuffle=False, drop_last=True)

era5_new_sample_ = next(iter(era5_new_dl_))
print(era5_new_sample_.shape)
# -

plt.imshow(era5_new_ds_.inverse_dwd_trafo(era5_new_sample_.numpy()) [0,0,:,:])


plt.imshow(era5_new_sample_[0,0,:,:])
plt.show()


# # QM SSP585 GFDL at 64   (no longer needed, now directly done in QM ssp notebooK)

def transform_full_gfdl_to_1d_64(do_save=False):
    path = 'data/deltaQM_ssp585_revision.pth'
    data_ = torch.load(path).cpu().numpy()
    data_, mean, std = dwd_rv_rainrate_transform_no_clip(data_)                 # log(x+1) -> standardize 
    data_min = data_.min()
    data_max = data_.max()
    data_ = (data_ - data_.min()) / (data_.max() - data_.min())                 # bring data to [-1,1]
    data_ = data_ * 2 - 1 
    
    print("data_", data_.shape)
    data_rescaled = F.interpolate(torch.tensor(data_).unsqueeze(1), scale_factor=4, mode='bilinear')#.squeeze(1)
    print("data_rescaled.shape:",data_rescaled.shape)
    
    
    if do_save==True:
        print("saving")
        torch.save(data_rescaled, "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/qm_gfdl_ssp585_256_2020_2100_processed.nc")
        
    return data_rescaled, mean, std, data_min, data_max

# +
do_test_functions_gfdl_64 = True

if do_test_functions_gfdl_64 == True:
    data_rescaled, mean, std, minval, maxval = transform_full_gfdl_to_1d_64(do_save=False)
    print(mean, std, minval, maxval)


# -

# # SSP585 GFDL at 64

# +
def transform_full_gfdl_to_1d_64():
    path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/pr_day_GFDL-ESM4_ssp585_r1i1p1f1_gr1_ever_10th_year_1d.nc"
    data_ = xr.open_dataset(path, cache=True, chunks=None).pr.values*24*3600    # load data and trafo to (mm/day) 
    data_, mean, std = dwd_rv_rainrate_transform_no_clip(data_)                 # log(x+1) -> standardize 
    data_min = data_.min()
    data_max = data_.max()
    data_ = (data_ - data_.min()) / (data_.max() - data_.min())                 # bring data to [-1,1]
    data_ = data_ * 2 - 1 
    
    return data_, mean, std, data_min, data_max


def create_full_gfdl_to_1d_64(do_save=False):
    path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/pr_day_GFDL-ESM4_ssp585_r1i1p1f1_gr1_ever_10th_year_1d.nc"
    dataset_original = xr.open_dataset(path, cache=True, chunks=None)
    full_gfdl_ds = xr.Dataset()
    full_gfdl_ds['time'] = dataset_original['time']  
    full_gfdl_ds['pr'] = (('time', 'lat', 'lon'), data_rescaled)
    full_gfdl_ds.attrs = dataset_original.attrs
    
    if do_save==True:
        print("saving")
        full_gfdl_ds.to_netcdf("/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/gfdl_ssp585_1d_2020_2100_processed.nc")
    
    return full_gfdl_ds


# +
do_test_functions_gfdl_64 = True

if do_test_functions_gfdl_64 == True:
    data_rescaled, mean, std, minval, maxval = transform_full_gfdl_to_1d_64()
    create_full_gfdl_to_1d_64(do_save=False)
    print(mean, std, minval, maxval)


# -

# # SSP585 GFDL at 256

# +
def transform_full_gfdl_to_1d_64():
    path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/pr_day_GFDL-ESM4_ssp585_r1i1p1f1_gr1_ever_10th_year_1d.nc"
    data_ = xr.open_dataset(path, cache=True, chunks=None).pr.values*24*3600    # load data and trafo to (mm/day) 
    data_, mean, std = dwd_rv_rainrate_transform_no_clip(data_)                 # log(x+1) -> standardize 
    data_min = data_.min()
    data_max = data_.max()
    data_ = (data_ - data_.min()) / (data_.max() - data_.min())                 # bring data to [-1,1]
    data_ = data_ * 2 - 1 
    
    print("data_", data_.shape)
    data_rescaled = F.interpolate(torch.tensor(data_).unsqueeze(1), scale_factor=4, mode='bilinear').squeeze(1)
    print("data_rescaled.shape:",data_rescaled.shape)
    
    return data_rescaled, mean, std, data_min, data_max


def create_full_gfdl_to_1d_64(do_save=False):
    path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/pr_day_GFDL-ESM4_ssp585_r1i1p1f1_gr1_ever_10th_year_1d.nc"
    dataset_original = xr.open_dataset(path, cache=True, chunks=None)
    full_gfdl_ds = xr.Dataset()
    full_gfdl_ds['time'] = dataset_original['time']  
    full_gfdl_ds['pr'] = (('time', 'lat', 'lon'), data_rescaled)
    full_gfdl_ds.attrs = dataset_original.attrs
    
    if do_save==True:
        print("saving")
        full_gfdl_ds.to_netcdf("/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/gfdl_ssp_2020_2100_processed.nc")
    
    return full_gfdl_ds


# +
do_test_functions_gfdl_64 = True

if do_test_functions_gfdl_64 == True:
    data_rescaled, mean, std, minval, maxval = transform_full_gfdl_to_1d_64()
    create_full_gfdl_to_1d_64(do_save=False)
    print(mean, std, minval, maxval)


# -

# ## GFDL 64

# +
def transform_full_gfdl_to_1d_64():
    path = "data/model_data/pr_day_GFDL-ESM4_historical_r1i1p1f1_gr1_19500101-20141231_full_time.nc"
    data_ = xr.open_dataset(path, cache=True, chunks=None).pr.values*24*3600    # load data and trafo to (mm/day) 
    data_, mean, std = dwd_rv_rainrate_transform_no_clip(data_)                 # log(x+1) -> standardize 
    data_min = data_.min()
    data_max = data_.max()
    data_ = (data_ - data_.min()) / (data_.max() - data_.min())                 # bring data to [-1,1]
    data_ = data_ * 2 - 1 
    data_ = torch.from_numpy(data_[:,26:90,216:267])
    
    data_flipped = torch.flip(data_, dims=(0, 1))
    data_flipped = torch.flip(data_flipped, dims=[0])

    data_rescaled = np.zeros((data_flipped.shape[0], 64, 64))

    for i in range(data_flipped.shape[0]):
        data_rescaled[i] = rescale(data_flipped[i].numpy(), (1, 1.25), anti_aliasing=False)


    data_rescaled = torch.tensor(data_rescaled).float()
    print("data_rescaled.shape:",data_rescaled.shape)
    
    data_rescaled = F.interpolate(data_rescaled.unsqueeze(1), scale_factor=4, mode='bilinear').squeeze(1)
    
    
    return data_rescaled, mean, std, data_min, data_max


def create_full_gfdl_to_1d_64(do_save=False):
    path = "data/model_data/pr_day_GFDL-ESM4_historical_r1i1p1f1_gr1_19500101-20141231_full_time.nc"
    dataset_original = xr.open_dataset(path, cache=True, chunks=None)
    full_gfdl_ds = xr.Dataset()
    full_gfdl_ds['time'] = dataset_original['time']  
    full_gfdl_ds['pr'] = (('time', 'lat', 'lon'), data_rescaled)
    full_gfdl_ds.attrs = dataset_original.attrs
    
    if do_save==True:
        print("saving")
        full_gfdl_ds.to_netcdf("/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/gfdl_1950_2014.nc")
    
    return full_gfdl_ds


# +
do_test_functions_gfdl_64 = True

if do_test_functions_gfdl_64 == True:
    data_rescaled, mean, std, minval, maxval = transform_full_gfdl_to_1d_64()
    create_full_gfdl_to_1d_64(do_save=False)
    mean, std, minval, maxval


# -

def transform_full_gfdl_to_1d_256_raw():
    path = "data/model_data/pr_day_GFDL-ESM4_historical_r1i1p1f1_gr1_19500101-20141231_full_time.nc"
    data_ = xr.open_dataset(path, cache=True, chunks=None).pr.values*24*3600    # load data and trafo to (mm/day) 
    data_, mean, std = dwd_rv_rainrate_transform_no_clip(data_)                 # log(x+1) -> standardize 

    data_ = torch.from_numpy(data_[:,26:90,216:267])
    
    data_flipped = torch.flip(data_, dims=(0, 1))
    data_flipped = torch.flip(data_flipped, dims=[0])

    data_rescaled = np.zeros((data_flipped.shape[0], 64, 64))

    for i in range(data_flipped.shape[0]):
        data_rescaled[i] = rescale(data_flipped[i].numpy(), (1, 1.25), anti_aliasing=False)


    data_rescaled = torch.tensor(data_rescaled).float()
    print("data_rescaled.shape:",data_rescaled.shape)
    
    data_rescaled = F.interpolate(data_rescaled.unsqueeze(1), scale_factor=4, mode='bilinear').squeeze(1)
    
    
    return data_rescaled

data_rescaled = transform_full_gfdl_to_1d_256_raw()

test_data = data_rescaled[:3]
test_data.shape

# +
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch

def plot_images_with_captions_trafo_units(images_list, threshold, do_savefig=False):
    num_images = len(images_list)
    num_vars = len(images_list[0])

    # Calculate the minimum and maximum values for normalization across all images
    all_images = [image.squeeze().cpu().detach().numpy() for sublist in images_list for image in sublist]
    min_value = min([img.min() for img in all_images])
    max_value = max([img.max() for img in all_images])
    
    
    fig, axs = plt.subplots(num_images, num_vars + 1, figsize=(num_vars * 4 + 1, num_images * 4), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(left=0.02, right=0.9, top=0.98, bottom=0.02, wspace=0.2, hspace=0.2)
    
    for i in range(num_images):
        for j in range(num_vars):
            ax = axs[i, j]
            img_data = images_list[i][j].squeeze().cpu().detach().numpy()  
            
            ax.imshow(img_data, extent=[-90, -26, -64, 0], origin='upper', transform=ccrs.PlateCarree(),
                      vmin=min_value, vmax=max_value, cmap='RdBu')
            ax.coastlines(resolution='10m', color='black', linewidth=1, alpha=0.5)
            ax.add_feature(cfeature.LAND, color='lightgray')

            if i == 0:
                title_prefix = "GFDL - sample"
            elif i == 1:
                title_prefix = "QM- GFDL - sample"
            elif i == 2:
                title_prefix = "DM- GFDL - sample"
            elif i == 3:
                title_prefix = "ERA5 - sample"

            ax.set_title('{} {}'.format(title_prefix, j + 1))  
            ax.set_extent([-90, -26, -64, 0], crs=ccrs.PlateCarree())  

        # Add colorbar to the right of each row
        cbar_ax = fig.add_axes([0.92, axs[i, -1].get_position().y0, 0.02, axs[i, -1].get_position().height])
        cbar = fig.colorbar(axs[i, 0].images[0], cax=cbar_ax)
        cbar.set_label('Colorbar Label')
        axs[i, -1].axis('off')  # Turn off the axis for the colorbar
        
    if do_savefig:
        plt.savefig(do_savefig)
        print("Saving to", do_savefig)
    plt.show()

    
plot_images_with_captions_trafo_units([test_data
                                       ,test_data
                                       ,test_data
                                       ,test_data]
                                       ,threshold=1, do_savefig=None)


# -

# # GFDL to 256

# +
def transform_full_gfdl_to_1d_256():
    path = "data/model_data/pr_day_GFDL-ESM4_historical_r1i1p1f1_gr1_19500101-20141231_full_time.nc"
    data_ = xr.open_dataset(path, cache=True, chunks=None).pr.values*24*3600    # load data and trafo to (mm/day) 
    data_, mean, std = dwd_rv_rainrate_transform_no_clip(data_)                 # log(x+1) -> standardize 
    data_min = data_.min()
    data_max = data_.max()
    data_ = (data_ - data_.min()) / (data_.max() - data_.min())                 # bring data to [-1,1]
    data_ = data_ * 2 - 1 
    data_ = torch.from_numpy(data_[:,26:90,216:267])
    
    data_flipped = torch.flip(data_, dims=(0, 1))
    data_flipped = torch.flip(data_flipped, dims=[0])

    data_rescaled = np.zeros((data_flipped.shape[0], 64, 64))

    for i in range(data_flipped.shape[0]):
        data_rescaled[i] = rescale(data_flipped[i].numpy(), (1, 1.25), anti_aliasing=False)


    data_rescaled = torch.tensor(data_rescaled).float()
    print("data_rescaled.shape:",data_rescaled.shape)
    
    data_rescaled = F.interpolate(data_rescaled.unsqueeze(1), scale_factor=4, mode='bilinear').squeeze(1)
    
    
    return data_rescaled, mean, std, data_min, data_max


def create_full_gfdl_to_1d_256(do_save=False):
    path = "data/model_data/pr_day_GFDL-ESM4_historical_r1i1p1f1_gr1_19500101-20141231_full_time.nc"
    dataset_original = xr.open_dataset(path, cache=True, chunks=None)
    full_gfdl_ds = xr.Dataset()
    full_gfdl_ds['time'] = dataset_original['time']  
    full_gfdl_ds['pr'] = (('time', 'lat', 'lon'), data_rescaled)
    full_gfdl_ds.attrs = dataset_original.attrs
    
    if do_save==True:
        print("saving")
        full_gfdl_ds.to_netcdf("/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/256_gfdl_1950_2014.nc")
    
    
    
    return full_gfdl_ds

# +
do_test_functions_gfdl_256 = True

if do_test_functions_gfdl_256 == True:
    data_rescaled, mean, std, minval, maxval = transform_full_gfdl_to_1d_256()
    create_full_gfdl_to_1d_256(do_save=True)
    mean, std, minval, maxval


# -

class gfdl_eval_256(data.Dataset):
    def __init__(self, stage):
    
        self.splits = {
            "train":  ("1992-01-01", "2011-01-01"), 
            "valid": ("2011-01-01", "2014-12-01"),  
            "1950_2014": ("1950-01-01", "2014-12-31"),
        }

        self.era5_path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/256_gfdl_1950_2014.nc"
        
        self.stage = stage
        self.era5 = None

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None)
        era5 = era5.sel(time=slice(self.splits[self.stage][0], self.splits[self.stage][1]))
        return era5

    def __getitem__(self, index):
        self.era5 = self.load_era5_data()
        return torch.unsqueeze(torch.tensor(self.era5.pr.values[index, :, :]), 0)

    def __len__(self):
        self.era5 = self.load_era5_data()
        return len(self.era5.time.values)
    
    def inverse_dwd_trafo(self, transformed_data):
        mean=0.3138951
        std=0.35952568
        min_value=-0.8730812
        max_value= 6.3121753
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        x *= std
        x += mean
        x = 10 ** x
        x -=1
        return x


# +
era5_new_ds_ = gfdl_eval(stage='train')


era5_new_dl_ = data.DataLoader(era5_new_ds_, batch_size=10, shuffle=False, drop_last=True)

era5_new_sample_ = next(iter(era5_new_dl_))
print(era5_new_sample_.shape)
# -

plt.imshow(era5_new_sample_[0,0,:,:])
plt.show()

# +
era5_new_ds = gfdl_eval_256(stage='train')


era5_new_dl = data.DataLoader(era5_new_ds, batch_size=10, shuffle=False, drop_last=True)

era5_new_sample = next(iter(era5_new_dl))
print(era5_new_sample.shape)

# +
plt.imshow(era5_new_sample[0,0,:,:])
plt.show()

plt.imshow(era5_new_sample_[0,0,:,:])
plt.show()


# -

# ## ERA5 1d at 256pixels

# +
def transform_era5_1d_256():
    path = "data/model_data/era5_daymean_1992_2014.nc"
    data_ = xr.open_dataset(path, cache=True, chunks=None).tp*1000*24      # load data and trafo to (mm/day) 
    
    resized_data = []

    for i in range(data_.shape[0]):
        original_image = data_[i, :, :]
        resized_data_ = original_image[::4,:][:,::4]
        resized_data.append(resized_data_)

    resized_era5_data = np.stack(resized_data)

    data_resize = xr.DataArray(resized_era5_data, dims=('time', 'x', 'y'), coords={'time': data_.time})
    data_resize.attrs = data_.attrs 
    

    data_, mean, std = dwd_rv_rainrate_transform_no_clip(data_resize.values)      # log(x+1) -> standardize 
    data_min = data_.min()
    data_max = data_.max()
    data_ = (data_ - data_.min()) / (data_.max() - data_.min())            # bring data to [-1,1]
    data_ = data_ * 2 - 1 
    
    data_ = F.interpolate(torch.tensor(data_).unsqueeze(1), scale_factor=4, mode='bilinear').squeeze(1)
    
    return data_, mean, std, data_min, data_max


def create_full_era5_1d_256(data, do_save):
    path = "data/model_data/era5_daymean_1992_2014.nc"
    dataset_original = xr.open_dataset(path, cache=True, chunks=None)
    full_gfdl_ds = xr.Dataset()
    full_gfdl_ds['time'] = dataset_original['time']  
    full_gfdl_ds['pr'] = (('time', 'lat', 'lon'), data)
    full_gfdl_ds.attrs = dataset_original.attrs
    
    if do_save==True:
        print("saving")
        full_gfdl_ds.to_netcdf("/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/era5_upscaled_1d_256_1992_2014.nc")
    
    return full_gfdl_ds


# +
do_test_functions_era5_1_256 = False

if do_test_functions_era5_1_256 == True:
    data_rescaled, mean, std, data_min, data_max = transform_era5_1d_256()
    create_full_era5_1d_256(data_rescaled, do_save=False)
    data_rescaled.shape, mean, std, data_min, data_max


# -

# # ERA5 0.25d at 256

# +
def transform_era5_0_25d_256():
    path = "data/model_data/era5_daymean_1992_2014.nc"
    data_ = xr.open_dataset(path, cache=True, chunks=None).tp.values*1000*24      # load data and trafo to (mm/day) 
    data_, mean, std = dwd_rv_rainrate_transform_no_clip(data_)      # log(x+1) -> standardize 
    data_min = data_.min()
    data_max = data_.max()
    data_ = (data_ - data_.min()) / (data_.max() - data_.min())            # bring data to [-1,1]
    data_ = data_ * 2 - 1 
    
    return data_, mean, std, data_min, data_max


def create_full_era5_0_25d_256(data, do_save):
    path = "data/model_data/era5_daymean_1992_2014.nc"
    dataset_original = xr.open_dataset(path, cache=True, chunks=None)
    full_gfdl_ds = xr.Dataset()
    full_gfdl_ds['time'] = dataset_original['time']  
    full_gfdl_ds['pr'] = (('time', 'lat', 'lon'), data)
    full_gfdl_ds.attrs = dataset_original.attrs
    
    if do_save==True:
        print("saving")
        full_gfdl_ds.to_netcdf("/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/era5_0_25d_256_1992_2014.nc")
    
    return full_gfdl_ds


# +
do_test_functions_era5_0_25_256 = True

if do_test_functions_era5_0_25_256 == True:
    data_rescaled, mean, std, data_min, data_max = transform_era5_0_25d_256()
    create_full_era5_0_25d_256(data_rescaled, do_save=False)
    print(data_rescaled.shape, mean, std, data_min, data_max)

# +
era5_new_ds = era5_0_25d_256(stage='train')


era5_new_dl = data.DataLoader(era5_new_ds, batch_size=10, shuffle=False, drop_last=True)

era5_new_sample = next(iter(era5_new_dl))
print(era5_new_sample.shape)
# -

plt.imshow(era5_new_sample[0,0,:,:])
plt.show()
plt.imshow(sample_era5_025_tr[0,0,:,:])
plt.show()


class original_era5(torch.utils.data.Dataset):
    ## data dim = mm/day 
    def __init__(self, stage):
        self.size_x = 64
        self.size_y = 64
    
        self.splits = {
                "train":  ("1992-01-01", "2011-01-01"),      
                "valid":  ("2011-01-01", "2014-12-01"),  
                "test":   (6939, 10226),}

        self.era5_path = "data/model_data/era5_daymean_1992_2014.nc"
        
        self.stage = stage
        # 1. load the data and transform it to mm/h
        self.era5 = self.load_era5_data()
    
        #self.era5.values = np.clip(self.era5.values,0,128)
        
        
        self.era5 = self.era5.sel(time=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
        print("dataset size",self.era5.values.shape)
        self.num_samples = len(self.era5.time.values)
        


    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).tp*1000*24 #bc data was m/h 
                                                                    #-> averaged still m/h -> *1000/24 = mm/day
        #print("era5", type(era5),era5.shape)
        resized_era5 = []

        for i in range(era5.shape[0]):
            original_image = era5[i, :, :]
            resized_era5_ = original_image[::4,:][:,::4]
            resized_era5.append(resized_era5_)

        resized_era5_data = np.stack(resized_era5)
        
        resized_era5_da = xr.DataArray(resized_era5_data, dims=('time', 'x', 'y'), coords={'time': era5.time})
        resized_era5_da.attrs = era5.attrs  # Copy attributes
        
    
        print("era5", type(era5),resized_era5_da.shape)
        return resized_era5_da #era5
    


    def __getitem__(self, index):   
        x = torch.from_numpy(self.era5.isel(time=index).values)[:self.size_x,:self.size_y].float().unsqueeze(0)
        return x 
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].numpy()[0]
        return data_array


class ERA5_P_0_25_to_1_downsampling_interpolate_upsample_256(torch.utils.data.Dataset):
    """
    * load in cm & obs data (mm/day)
    * transform the data to mm/h and clip it between [0,128]
    * quantize the data to 1/32
    * log trafo with threshold + smoothing + Normalize the data m=0, std=1     
    * bring data to [-1,1]
    
    # load 0.25d data (m/day)? -> trafo to mm/h ?
    # taking every 4th datapoint in x,y dimension -> 1d data
    # just copy every datapoint 4 times -> 64x64 -> 256x256 pixel outputs
    """
    ## data dim = mm/day 
    def __init__(self, stage):
        self.size_x = 256
        self.size_y = 256
    
        self.splits = {
                "train":  ("1992-01-01", "2011-01-01"),      
                "valid":  ("2011-01-01", "2014-12-01"),  
                "test":   (6939, 10226),}

        self.era5_path = "data/model_data/era5_daymean_1992_2014.nc"
        
        self.stage = stage
        # 1. load the data and transform it to mm/h
        self.era5 = self.load_era5_data()
        self.era5.values, self.mean, self.std = dwd_rv_rainrate_transform_no_clip(self.era5.values)
        
        # 5. trafo to [-1,1]
        min_value = self.era5.values.min()
        max_value = self.era5.values.max()
        self.min_value = min_value
        self.max_value = max_value
        self.era5.values = (self.era5.values - min_value) / (max_value - min_value)  
        self.era5.values = self.era5.values * 2 - 1 

        self.era5 = self.era5.sel(time=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
        print("dataset size",self.era5.values.shape)
        self.num_samples = len(self.era5.time.values)
        

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).tp*1000*24 #bc data was m/h 
                                                                    #-> averaged still m/h -> *1000/24 = mm/day
        #print("era5", type(era5),era5.shape)
        resized_era5 = []

        for i in range(era5.shape[0]):
            original_image = era5[i, :, :]
            resized_era5_ = original_image[::4,:][:,::4]
            resized_era5.append(resized_era5_)

        resized_era5_data = np.stack(resized_era5)
        
        resized_era5_da = xr.DataArray(resized_era5_data, dims=('time', 'x', 'y'), coords={'time': era5.time})
        resized_era5_da.attrs = era5.attrs  # Copy attributes
        
    
        #print("era5", type(era5),resized_era5_da.shape)
        return resized_era5_da #era5
    
    
    
    def get_mean_std(self):
        print("mean:",self.mean)
        print("std:",self.std)
        print("min:",self.min_value)
        print("max:",self.max_value)
        return self.mean, self.std, self.min_value, self.max_value
        

    def __getitem__(self, index):   
        x = torch.from_numpy(self.era5.isel(time=index).values)[:self.size_x,:self.size_y].float().unsqueeze(0).unsqueeze(1)
        x = F.interpolate(x, scale_factor=4, mode='nearest').squeeze(1)
        return x 
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].numpy()[0]
        return data_array
    def inverse_dwd_trafo(self, transformed_data):
        mean=self.mean
        std=self.std 
        min_value=self.min_value
        max_value=self.max_value
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        # Inverse standardization
        x *= std
        x += mean
        # Inverse log 
        x = 10 ** x
        x -=1
        return x


class BC_GFDL_Dataset_256(torch.utils.data.Dataset):
    """ 
    first load original era5 data at 64x64 (1d) to get mean,std,min,max to do reverse trafo on bias corr. gfdl data
    do inverse trafo with the statisics of era5.. bc data should be similar to era5
    """
    ## data dim = mm/day 
    def __init__(self, path_bc_ds='data/bias_corr_gfdl_dataset.pth'):
        self.size_x = 256
        self.size_y = 256
        
        self.path_bc_ds = path_bc_ds

        self.bc_gfdl = torch.load(path_bc_ds)
        print("dataset size",self.bc_gfdl.shape)
        self.num_samples = self.bc_gfdl.shape[0]


    def __getitem__(self, index):   
        x = self.bc_gfdl[index].unsqueeze(1)
        x = F.interpolate(x, scale_factor=4, mode='nearest').squeeze(1)
        return x
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].cpu().numpy()[0]
        return data_array


class SR_BC_GFDL_Dataset_256(torch.utils.data.Dataset):
    """ 
    first load original era5 data at 64x64 (1d) to get mean,std,min,max to do reverse trafo on bias corr. gfdl data
    do inverse trafo with the statisics of era5.. bc data should be similar to era5
    """
    ## data dim = mm/day 
    def __init__(self, path='data/HR_BC_GFDL_dataset.pth'):
        self.size_x = 256
        self.size_y = 256

        self.path = path

        self.hr_bc_gfdl = torch.load(self.path)
        print("dataset size",self.hr_bc_gfdl.shape)
        self.num_samples = self.hr_bc_gfdl.shape[0]

    def __getitem__(self, index):   
        x = self.hr_bc_gfdl[index]
        return x
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].cpu().numpy()[0]
        return data_array


class SR_ERA5_HR_Dataset_256(torch.utils.data.Dataset):
    """ 
    first load original era5 data at 64x64 (1d) to get mean,std,min,max to do reverse trafo on bias corr. gfdl data
    do inverse trafo with the statisics of era5.. bc data should be similar to era5
    """
    ## data dim = mm/day 
    def __init__(self, path= 'data/SR_ERA5_HR_dataset.pth'):
        self.size_x = 256
        self.size_y = 256
        self.path = path

        
        self.hr_bc_gfdl = torch.load(self.path)
        print("datasets size",self.hr_bc_gfdl.shape)
        self.num_samples = self.hr_bc_gfdl.shape[0]
        

    def __getitem__(self, index):   
        x = self.hr_bc_gfdl[index]
        return x
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].cpu().numpy()[0]
        return data_array
