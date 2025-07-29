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



from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, folder_path, 
                 n_files=3, 
                 mode='last',
                 max_items=None):
        """
        Initialize the dataset by loading only a subset of files
        and optionally limiting the total number of items (e.g. years).

        Args:
            folder_path (str): Path to the folder containing .pth files.
            n_files (int): How many files to load from the folder 
                           (ignored if mode='all').
            mode (str): 'first', 'last', or 'all'.
            max_items (int): If not None, limit the total number of data points
                             from the selected files to at most this value.
        """
        self.folder_path = folder_path
        self.max_items = max_items

        # Gather all .pth files
        all_file_names = [
            f for f in os.listdir(folder_path) if f.endswith('.pth')
        ]
        
        # Sort the file names so "first" or "last" is consistent/predictable
        all_file_names.sort()

        # Select files based on the mode
        if mode == 'first':
            selected_file_names = all_file_names[:n_files]
        elif mode == 'last':
            selected_file_names = all_file_names[-n_files:]
        elif mode == 'all':
            selected_file_names = all_file_names  # Load all files
        else:
            raise ValueError("mode must be 'first', 'last', or 'all'.")

        # Convert file names to full paths
        self.file_paths = [os.path.join(folder_path, f) for f in selected_file_names]
        
        # Calculate how many items are in each file (and handle partial if needed)
        self.element_counts = self._calculate_element_counts()

        # Create the cumulative sums for indexing
        self.cumulative_counts = self._calculate_cumulative_counts()
        
        # Placeholders for caching loaded files
        self.current_file_data = None
        self.current_file_index = -1

    def _calculate_element_counts(self):
        """
        Calculate the number of elements in each of the selected files,
        optionally respecting self.max_items to limit total.

        Returns:
            list[int]: A list with the number of elements in each file 
                       (possibly reduced for the last partial file if max_items is set).
        """
        element_counts = []
        total_so_far = 0
        
        for file_path in self.file_paths:
            data = torch.load(file_path, weights_only=True)
            num_in_file = len(data)

            if self.max_items is not None:
                # How many items can we still load without exceeding max_items?
                can_take = self.max_items - total_so_far
                if can_take <= 0:
                    # We already have enough items; stop loading more files
                    break
                elif can_take < num_in_file:
                    # Need only a partial chunk from this file
                    element_counts.append(can_take)
                    total_so_far += can_take
                    break  # Stop after partial
                else:
                    # We can take the whole file
                    element_counts.append(num_in_file)
                    total_so_far += num_in_file
            else:
                # No max limit, take entire file
                element_counts.append(num_in_file)
        
        return element_counts

    def _calculate_cumulative_counts(self):
        """
        Compute cumulative sums of element counts to determine file boundaries.

        Returns:
            list[int]: Cumulative boundary indices for each file.
        """
        cumulative = [0]
        for count in self.element_counts:
            cumulative.append(cumulative[-1] + count)
        return cumulative

    def _load_file(self, file_index):
        """
        Load data from the specified file index if it's not already loaded.

        Args:
            file_index (int): Index of the file to load.

        Returns:
            Any: Data from the loaded file.
        """
        if self.current_file_index != file_index:
            file_path = self.file_paths[file_index]
            self.current_file_data = torch.load(file_path, weights_only=True)
            self.current_file_index = file_index
        return self.current_file_data

    def __getitem__(self, index):
        """
        Get the item at the specified index.

        Args:
            index (int): Index of the data item.

        Returns:
            Any: The data item at the specified index.
        """
        # Determine which file contains the `index`
        for file_index, (start, end) in enumerate(zip(self.cumulative_counts[:-1], 
                                                      self.cumulative_counts[1:])):
            if start <= index < end:
                # Load the file data
                file_data = self._load_file(file_index)
                local_index = index - start
                return file_data[local_index]
        
        raise IndexError("Index out of range")

    def __len__(self):
        """
        Return the total number of items in the dataset.

        Returns:
            int: Total number of items in the dataset.
        """
        return self.cumulative_counts[-1]





### Dataloader original units (with fw/bw trafo) GFDL ssp585 after QM ###

class GFDL_P_Dataset_64_ssp_2015_2100_original_unit_after_QM(torch.utils.data.Dataset):

    def __init__(self):
        self.size_x = 64
        self.size_y = 64
    

        self.era5_path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/pr_QDM_GFDL_ssp585_r1i1p1f1_original_units_2015_2100.pth"
        self.era5 = None

        # from LR ERA5 (to match embedding alignment)
        self.mean=0.37070283
        self.std=0.39181525
        self.min_value = -0.9461164
        self.max_value = 5.897021

    def __getitem__(self, index):
        if self.era5 is None:
            self.era5 = torch.load(self.era5_path)
        return torch.tensor(self.era5[index, :, :]).unsqueeze(0)

    def __len__(self):
        if self.era5 is None:
            self.era5 = torch.load(self.era5_path)
        return self.era5[:,0,0].shape[0]

    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].numpy()[0]
        return data_array


    def fw_trafo(self, x):
        """
        Forward transformation: Convert original values into a normalized range of [-1, 1].
        This function reverses the steps in the inverse transformation.
        """
        # Reverse the inverse transformation:
        #   Inverse: x = ((x + 1)/2 * (max - min) + min) * std + mean
        # Forward:
        x = x + 1
        x = np.log10(x)
        x = (x - self.mean) / self.std                         # Undo mean and std scaling
        x = (x - self.min_value) / (self.max_value - self.min_value)  # Undo scaling to [0, 1]
        x = x * 2 - 1                                          # Scale from [0,1] to [-1,1]
        return x

    def bw_trafo(self, x):
        """
        Inverse transformation: Convert normalized [-1, 1] data back to original units.
        """
        x = (x + 1) / 2
        x = x * (self.max_value - self.min_value) + self.min_value
        x = x * self.std
        x = x + self.mean
        x = 10 ** x
        x -= 1
        return x

### Dataloader original units GFDL ssp585 before QM ###
class GFDL_P_Dataset_64_ssp_2015_2100_original_unit_before_QM(torch.utils.data.Dataset):

    def __init__(self, stage):
        self.size_x = 64
        self.size_y = 64
    
        self.splits = {
                "train":  ("2015-01-01", "2100-12-31"),      
                "valid":  ("2011-01-01", "2014-12-01"),  ### mistake everywhere 2014-12-31
                "1950_2014":   ("1950-01-01", "2014-12-01"),
                "ssp370":  ("2015-01-01", "2100-12-31"),
                "2081_2100":  ("2081-01-01", "2100-12-31"),}

        self.era5_path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/pr_day_GFDL-ESM4_ssp585_r1i1p1f1_gr1_2015_2100.nc"

        self.stage = stage
        # 1. load the data and transform it to mm/h
        self.era5 = self.load_era5_data()

        self.era5 = self.era5.sel(time=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
        print("dataset size",self.era5.values.shape)
        self.num_samples = len(self.era5.time.values)
        

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).pr*24*3600
        return era5


    def __getitem__(self, index):   
        x = torch.from_numpy(self.era5[index,26:90,216:267].values)
        flipped_x = torch.flip(x, dims=(0, 1))
        gfdl = torch.flip(flipped_x, dims=[1])
        
        gfdl_rescale = np.zeros((64, 64))
        gfdl_rescale = rescale(gfdl, scale=(1, 1.25), anti_aliasing=False)
        
        return torch.tensor(gfdl_rescale).float().unsqueeze(0)
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].numpy()[0]
        return data_array





### QM of 1995-2014 gfdl ###

class GFDL_P_Dataset_64_ssp_1995_2014_original_unit_after_QM(torch.utils.data.Dataset):

    def __init__(self):
        self.size_x = 64
        self.size_y = 64
    
        self.era5_path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/pr_QDM_GFDL_1995_2014_r1i1p1f1_original_units.pth"
        self.era5 = None

        # from LR ERA5 (to match embedding alignment)
        self.mean=0.37070283
        self.std=0.39181525
        self.min_value = -0.9461164
        self.max_value = 5.897021
        
    def __getitem__(self, index):
        if self.era5 is None:
            self.era5 = torch.load(self.era5_path)
        return torch.tensor(self.era5[index, :, :]).unsqueeze(0)

    def __len__(self):
        if self.era5 is None:
            self.era5 = torch.load(self.era5_path)
        return self.era5[:,0,0].shape[0]

    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].numpy()[0]
        return data_array


    def fw_trafo(self, x):
        """
        Forward transformation: Convert original values into a normalized range of [-1, 1].
        This function reverses the steps in the inverse transformation.
        """
        # Reverse the inverse transformation:
        #   Inverse: x = ((x + 1)/2 * (max - min) + min) * std + mean
        # Forward:
        x = x + 1
        x = np.log10(x)
        x = (x - self.mean) / self.std                         # Undo mean and std scaling
        x = (x - self.min_value) / (self.max_value - self.min_value)  # Undo scaling to [0, 1]
        x = x * 2 - 1                                          # Scale from [0,1] to [-1,1]
        return x

    def bw_trafo(self, x):
        """
        Inverse transformation: Convert normalized [-1, 1] data back to original units.
        """
        x = (x + 1) / 2
        x = x * (self.max_value - self.min_value) + self.min_value
        x = x * self.std
        x = x + self.mean
        x = 10 ** x
        x = x - 1
        return x



# # QM then bi-linear downscale GFDL - validation   (part of embedding )

# +
class QM_GFDL_LR_Dataset_256(torch.utils.data.Dataset):
    """
    * load in cm & obs data (mm/day)
    * log trafo with threshold + smoothing + Normalize the data m=0, std=1     
    * bring data to [-1,1]
    """
    ## data dim = mm/day 
    def __init__(self, path='data/QM_gfdl_lr_64_training_dataset.pth'):
        self.size_x = 256
        self.size_y = 256
        
        self.path = path
        self.era5_qm = torch.load(self.path).cpu().numpy()
        
        self.era5_qm, self.mean_qm, self.std_qm, self.min_value_qm,self.max_value_qm = dwd_rv_rainrate_transform_no_clip_new_era5_proc(self.era5_qm)
        

        self.era5_qm = (self.era5_qm - self.min_value_qm) / (self.max_value_qm - self.min_value_qm)  
        self.era5_qm = self.era5_qm * 2 - 1 
        self.era5_qm = torch.tensor(self.era5_qm)
        
        print("datasets size",self.era5_qm.shape)
        self.num_samples = self.era5_qm.shape[0]
        
    
    def __getitem__(self, index):
        if self.era5_qm is None:
            self.era5_qm = torch.load(self.path)
        x = self.era5_qm[index].unsqueeze(0).unsqueeze(1)
        x = F.interpolate(x, scale_factor=4, mode='bilinear').squeeze(1)
        return x
        
    
    def __len__(self):
        if self.era5_qm is None:
            self.era5_qm = torch.load(self.path)
        return self.era5_qm.shape[0]
    
    
    def inverse_dwd_trafo(self, transformed_data):
        mean=0.37120456
        std=0.39180517
        min_value=-0.94742125
        max_value=5.895891
        
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        x *= std
        x += mean
        x = 10 ** x
        x -=1
        return x
    
    
def dwd_rv_rainrate_transform_no_clip_new_era5_proc(raw, mu=0.37120456,std=0.39180517,
                                                    min_v=-0.94742125, max_v=5.895891):
    # this is from LR ERA5 (mean,std,min,max)  (as QM was fit to match lr gfdl to lr era5 )
    
    # 1. qm (data will be similar to lr era5) 
    # 2. trafo qm with lr era5 mu,std,min,max 
    # 3. trafo qm back with lr era 5 mu,std,min,max 
    # 4. upscaling
    
    x = raw.copy()
    x += 1
    x = np.log10(x, out=x)

    x -= mu       
    x /= std
    
    return x, mu, std, min_v, max_v




# # bi-linear downscale GFDL then QM DS (original units) - for Benchmark

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


# # SSP585 LR GFDL to 64

class gfdl_eval_ssp585_1d(data.Dataset):
    def __init__(self):
    
        self.era5_path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/gfdl_ssp585_1d_2020_2100_processed.nc"
        self.era5 = None

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None)
        return era5

    def __getitem__(self, index):
        if self.era5 is None:
            self.era5 = self.load_era5_data()
        return torch.unsqueeze(torch.tensor(self.era5.pr.values[index, :, :]), 0)

    def __len__(self):
        if self.era5 is None:
            self.era5 = self.load_era5_data()
        return len(self.era5.time.values)
    
    def inverse_dwd_trafo(self, transformed_data):
        mean=0.35393184  
        std=0.38565448 
        min_value=-0.9177434
        max_value= 5.3501124
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        x *= std
        x += mean
        x = 10 ** x
        x -=1
        return x


# # SSP585 LR GFDL to 256

class gfdl_eval_ssp585(data.Dataset):
    def __init__(self):
    
        self.era5_path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/gfdl_ssp_2020_2100_processed.nc"
        self.era5 = None

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None)
        return era5

    def __getitem__(self, index):
        if self.era5 is None:
            self.era5 = self.load_era5_data()
        return torch.unsqueeze(torch.tensor(self.era5.pr.values[index, :, :]), 0)

    def __len__(self):
        if self.era5 is None:
            self.era5 = self.load_era5_data()
        return len(self.era5.time.values)
    
    def inverse_dwd_trafo(self, transformed_data):
        mean=0.35393184   
        std=0.38565448
        min_value=-0.9177434
        max_value= 5.3501124
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        x *= std
        x += mean
        x = 10 ** x
        x -=1
        return x


# # QM ssp585 GFDL

class QM_gfdl_256_ssp585(data.Dataset):
    def __init__(self, stage, path = "/dss/dsshome1/0D/ge74xuf2/climate_diffusion/data/deltaQM_ssp585_revision_trafo_unit.pth" ):
    
        self.splits = {
            "train":  (0, 3285), 
            "valid": (0, 3285),  
        }

        self.era5_path = path
        
        self.stage = stage
        self.era5 = None

    def load_era5_data(self):
        era5 = torch.load(self.era5_path)
        start, end = self.splits[self.stage]
        era5_train_valid = era5[start:end]
        
        era5_train_valid = F.interpolate(era5_train_valid, scale_factor=4, mode='bilinear')
        return era5_train_valid

    def __getitem__(self, index):
        if self.era5 is None:
            self.era5 = self.load_era5_data()
        
        return self.era5[index, :, :, :]

    def __len__(self):
        if self.era5 is None:
            self.era5 = self.load_era5_data()
        return len(self.era5)
    
    def inverse_dwd_trafo(self, transformed_data):
        mean=0.37070283
        std=0.39181525
        min_value= -0.9461164
        max_value= 5.897021
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        x *= std
        x += mean
        x = 10 ** x
        x -=1
        return x



# # GFDL LR to 256

class gfdl_eval_256(data.Dataset):
    def __init__(self, stage):
    
        self.splits = {
            "train":  ("1992-01-01", "2011-01-01"), 
            "valid": ("2011-01-01", "2014-12-01"),  
            "1950_2014": ("1950-01-01", "2014-12-31"),
            "1995_2014": ("1995-01-01", "2014-12-31"),
        }

        self.era5_path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/256_gfdl_1950_2014.nc"
        
        self.stage = stage
        self.era5 = None

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None)
        era5 = era5.sel(time=slice(self.splits[self.stage][0], self.splits[self.stage][1]))
        return era5

    def __getitem__(self, index):
        if self.era5 is None:
            self.era5 = self.load_era5_data()
        return torch.unsqueeze(torch.tensor(self.era5.pr.values[index, :, :]), 0)

    def __len__(self):
        if self.era5 is None:
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


# # GFDL LR 64

class gfdl_eval(data.Dataset):
    def __init__(self, stage):
    
        self.splits = {
            "train":  ("1992-01-01", "2011-01-01"), 
            "valid": ("2011-01-01", "2014-12-01"),  
            "1950_2014": ("1950-01-01", "2014-12-31"),
            "1995_2014": ("1995-01-01", "2014-12-31"),
        }

        self.era5_path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/gfdl_1950_2014.nc"
        
        self.stage = stage
        self.era5 = None

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None)
        era5 = era5.sel(time=slice(self.splits[self.stage][0], self.splits[self.stage][1]))
        return era5

    def __getitem__(self, index):
        if self.era5 is None:
            self.era5 = self.load_era5_data()
        return torch.unsqueeze(torch.tensor(self.era5.pr.values[index, :, :]), 0)

    def __len__(self):
        if self.era5 is None:
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


# # ERA5 LR to 1d 256

class era5_upscaled_1d_256(data.Dataset):
    def __init__(self, stage):
        self.splits = {
            "train":  ("1992-01-01", "2011-01-01"),      
            "valid":  ("2011-01-02", "2014-12-01"),  
            "test":   (6939, 10226),
        }
        self.era5_path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/era5_upscaled_1d_256_1992_2014.nc"
        self.stage = stage
        self.era5 = None

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None)
        era5 = era5.sel(time=slice(self.splits[self.stage][0], self.splits[self.stage][1]))
        return era5

    def __getitem__(self, index):
        if self.era5 is None:
            self.era5 = self.load_era5_data()
        return torch.unsqueeze(torch.tensor(self.era5.pr.values[index, :, :]), 0)

    def __len__(self):
        if self.era5 is None:
            self.era5 = self.load_era5_data()
        return len(self.era5.time.values)
    
    def inverse_dwd_trafo(self, transformed_data):
        mean=0.37070283
        std=0.39181525
        min_value=-0.9461164
        max_value= 5.897021
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        x *= std
        x += mean
        x = 10 ** x
        x -=1
        return x


# # ERA5 HR at 0.25 256

class era5_0_25d_256(data.Dataset):
    def __init__(self, stage):
        self.splits = { "train":  ("1992-01-01", "2011-01-01"),      
                        "valid":  ("2011-01-02", "2014-12-01"),  
                        "test":   ("1992-01-01", "2014-12-01"),}

        self.era5_path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/era5_0_25d_256_1992_2014.nc"
        self.stage = stage
        self.era5 = None

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None)
        era5 = era5.sel(time=slice(self.splits[self.stage][0], self.splits[self.stage][1]))
        return era5

    def __getitem__(self, index):
        if self.era5 is None:
            self.era5 = self.load_era5_data()
        return torch.unsqueeze(torch.tensor(self.era5.pr.values[index, :, :]), 0)

    def __len__(self):
        if self.era5 is None:
            self.era5 = self.load_era5_data()
        return len(self.era5.time.values)
    
    def inverse_dwd_trafo(self, transformed_data):
        mean=0.3700341
        std=0.39072984
        min_value=-0.9470331
        max_value= 6.000293
            
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        x *= std
        x += mean
        x = 10 ** x
        x -=1
        return x


# # dataloader for DM-corrected GFDL 

class SR_BC_GFDL_Dataset_256(torch.utils.data.Dataset):
    """ 
    first load original era5 data at 64x64 (1d) to get mean,std,min,max to do reverse trafo on bias corr. gfdl data
    do inverse trafo with the statisics of era5.. bc data should be similar to era5
    """
    ## data dim = mm/day 
    def __init__(self, path='data/HR_BC_GFDL_dataset.pth'):
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


# # bi-linear then QM DS (era5 units) - does not work 

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


# # 1d 64pixel data for QM (produces inference GFDL data)

def dwd_rv_rainrate_transform_no_clip(raw, threshold=0.1, fill_value=0.001):
   
    x = raw.copy()
    x += 1                              #x[x < threshold] = fill_value
    x = np.log10(x, out=x)
                                        #epsilon = 0.01
                                        #x = np.log(x + epsilon) - np.log(epsilon)
    mu = np.mean(x)
    std = np.std(x)
    x -= mu       
    x /= std
    return x, mu, std


# +
class QM_GFDL_LR_Dataset_64(torch.utils.data.Dataset):
    """
    * load in cm & obs data (mm/day)
    * transform the data to mm/h and clip it between [0,128]
    * quantize the data to 1/32
    * log trafo with threshold + smoothing + Normalize the data m=0, std=1     
    * bring data to [-1,1]
    
    # load 0.25d data (m/day)? -> trafo to mm/h ?
    # taking every 4th datapoint in x,y dimension -> 1d data
    """
    ## data dim = mm/day 
    def __init__(self, path='data/QM_gfdl_lr_64_training_dataset.pth'):
        self.size_x = 64
        self.size_y = 64
        
        self.path = path
        self.era5_qm = torch.load(self.path).cpu().numpy()
        
        self.era5_qm, self.mean_qm, self.std_qm, self.min_value_qm,self.max_value_qm = dwd_rv_rainrate_transform_no_clip_new_era5_proc(self.era5_qm)
        

        self.era5_qm = (self.era5_qm - self.min_value_qm) / (self.max_value_qm - self.min_value_qm)  
        self.era5_qm = self.era5_qm * 2 - 1 
        self.era5_qm = torch.tensor(self.era5_qm)
        
        print("datasets size",self.era5_qm.shape)
        self.num_samples = self.era5_qm.shape[0]
        

    
    def get_mean_std(self):
        print("mean:",self.mean_qm)
        print("std:",self.std_qm)
        print("min:",self.min_value_qm)
        print("max:",self.max_value_qm)
        return self.mean_qm, self.std_qm, self.min_value_qm, self.max_value_qm
        

    def __getitem__(self, index):   
        x = self.era5_qm[index].unsqueeze(0)
        return x 
    
    def size(self):
        return self.num_samples
    
    def __len__(self):
        return self.num_samples
    
    def data(self):
        data_array = np.zeros((self.num_samples, self.size_x, self.size_y))
        for i in range(len(self)):
            data_array[i] = self[i].cpu().numpy()#[0]
        return data_array
    
    def inverse_dwd_trafo(self, transformed_data):
        #"""
        mean=self.mean_qm
        std=self.std_qm
        min_value=self.min_value_qm
        max_value=self.max_value_qm
        x = transformed_data
        x = (x + 1) / 2
        x = x * (max_value - min_value) + min_value
        x *= std
        x += mean
        x = 10 ** x
        x -=1
        return x
    
    
def dwd_rv_rainrate_transform_no_clip_new_era5_proc(raw, mu=0.37120456,std=0.39180517,
                                                    min_v=-0.94742125, max_v=5.895891):
    # this is from LR ERA5 (mean,std,min,max)  (as QM was fit to match lr gfdl to lr era5 )
    # 1. qm 2. trafo qm with its own mu,std,min,max 3. trafo qm back with lr era 5  , 4. US
    x = raw.copy()
    x += 1
    x = np.log10(x, out=x)

    x -= mu       
    x /= std
    
    return x, mu, std, min_v, max_v


# -

class ERA5_P_0_25_to_1_Dataset(torch.utils.data.Dataset):

    def __init__(self, stage):
        self.size_x = 64
        self.size_y = 64
    
        self.splits = {
                "train":  ("1992-01-01", "2011-01-01"),      
                "valid":  ("2011-01-01", "2014-12-01"),  
                "all":  ("1992-01-01", "2014-12-01"),  
                "test":   (6939, 10226),}

        self.era5_path = "data/model_data/era5_daymean_1992_2014.nc"
        
        self.stage = stage
        # 1. load the data and transform it to mm/h
        self.era5 = self.load_era5_data()

        
        # 4. log trafo with threshold + smoothing + Normalize the data m=0, std=1        
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
        
    
        print("era5", type(era5),resized_era5_da.shape)
        return resized_era5_da #era5
    
    
    def get_mean_std(self):
        print("mean:",self.mean)
        print("std:",self.std)
        print("min:",self.min_value)
        print("max:",self.max_value)
        return self.mean, self.std, self.min_value, self.max_value
        

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


class GFDL_P_Dataset_64_1992_2014(torch.utils.data.Dataset):
    """ from 2010-2014
    Data has lon-resolution: 1.25d, lat-resolution 1d , 
    lat, lon dimension: 26:90,216:267 (64,51) -> data spans the same region on the globe as era5
    data is interpolated to 64,64 pixels -> 1x1 degree
    
    * load in cm & obs data (mm/day)
    * transform the data to mm/h and clip it between [0,128]
    * quantize the data to 1/32
    * log trafo with threshold + smoothing + Normalize the data m=0, std=1     
    * bring data to [-1,1]
    """
    ## data dim = mm/day 
    def __init__(self, stage):
        self.size_x = 64
        self.size_y = 64
    
        self.splits = {
                #"train":  ("1950-01-01", "2008-12-31"),  
                "train":  ("1992-01-01", "2011-01-01"),      
                "valid":  ("2011-01-01", "2014-12-01"),  ### mistake everywhere 2014-12-31
                "1950_2014":   ("1950-01-01", "2014-12-31"),}

        self.era5_path = "data/model_data/pr_day_GFDL-ESM4_historical_r1i1p1f1_gr1_19500101-20141231_full_time.nc"
                self.era5_path = "/dss/dssfs04/lwp-dss-0002/pn49fu/pn49fu-dss-0002/ge74xuf2/diffusion/data/gfdl_1950_2014.nc"

        
        self.stage = stage
        # 1. load the data and transform it to mm/h
        self.era5 = self.load_era5_data()
    

        self.era5.values, self.mean, self.std = dwd_rv_rainrate_transform_no_clip(self.era5.values)
        
        
        # 5. trafo to [-1,1]
        self.min_value = self.era5.values.min()
        self.max_value = self.era5.values.max()
        self.era5.values = (self.era5.values - self.min_value) / (self.max_value - self.min_value)  
        self.era5.values = self.era5.values * 2 - 1 

        self.era5 = self.era5.sel(time=slice(self.splits[self.stage][0],self.splits[self.stage][1]))
        print("dataset size",self.era5.values.shape)
        self.num_samples = len(self.era5.time.values)
        

    def load_era5_data(self):
        era5 = xr.open_dataset(self.era5_path, cache=True, chunks=None).pr*24*3600
            
        return era5
    
    
    def get_mean_std(self):
        print("mean:",self.mean)
        print("std:",self.std)
        print("min:",self.min_value)
        print("max:",self.max_value)
        return self.mean, self.std, self.min_value, self.max_value
        

    def __getitem__(self, index):   
        x = torch.from_numpy(self.era5[index,26:90,216:267].values)
        flipped_x = torch.flip(x, dims=(0, 1))
        gfdl = torch.flip(flipped_x, dims=[1])
        
        gfdl_rescale = np.zeros((64, 64))
        gfdl_rescale = rescale(gfdl, scale=(1, 1.25), anti_aliasing=False)
        
        return torch.tensor(gfdl_rescale).float().unsqueeze(0)
    
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



class load_dataset(data.Dataset):
    def __init__(self, stage, path, batch_size=1,):
        self.splits = {
            "train":  (0, 14000),
            "valid": (14000, 16425),
            "test": (0, 8),
            "all":  (0, 16425),
            "ssp370":  (0, 31390),
            "last10y":  (27740, 31390),
            "first25y": (0, 9450),
            "first30y": (0, 10950),
            "first40y": (0, 14600),
            "last25yssp": (21900, 31350),
            "last25yhist": (6975, 16425),
        }
        self.era5_path = path
        self.stage = stage
        self.era5 = None
        self.batch_size = batch_size

    def load_era5_data(self):
        era5 = torch.load(self.era5_path, weights_only=True)
        era5 = era5[self.splits[self.stage][0]: self.splits[self.stage][1]]
        return era5

    def __getitem__(self, index):
        if self.era5 is None:
            self.era5 = self.load_era5_data()

        era5_slice = self.era5[index]
        era5_array = era5_slice
        return era5_array

    def __len__(self):
        if self.era5 is None:
            self.era5 = self.load_era5_data()
        return len(self.era5)

    def collate_fn(self, batch):
        return torch.stack(batch)
from torch.utils.data import  DataLoader
