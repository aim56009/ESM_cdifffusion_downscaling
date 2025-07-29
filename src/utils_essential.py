import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from inspect import isfunction

# %%capture
# !pip install pysteps

from pysteps.utils.spectral import rapsd, corrcoef
import matplotlib.ticker as ticker
from scipy.stats import wasserstein_distance


# # evaluate sr bc gfdl

class SpatialSpectralDensity_4_diff_res():
    """
    1 & 2 argument have a different resolution than argument 2&3 
    """
    def __init__(self, original, generated, label, comparison=None, 
                 new_labels = ["original","generated", "UNet", "unet"]
                 ,y_ax_name='PSD [a.u]',x_ax_name='Wavelength[km]'
                 ,plot_info_for_confused=False, title=None):
        
        self.original = original
        self.generated = generated  
        self.label = label  
        self.new_label_ori = new_labels[0]
        self.new_label_gen = new_labels[1]
        self.new_label_lab = new_labels[2]
        self.comparison = comparison
        self.new_label_comp = new_labels[3]
        self.x_ax_name = x_ax_name
        self.y_ax_name = y_ax_name
        self.plot_info_for_confused = plot_info_for_confused
        self.title = title
   
    def compute_mean_spectral_density(self, data):
        
        num_frequencies = np.max(((data.shape[3]),(data.shape[3])))/2
        mean_spectral_density = np.zeros(int(num_frequencies))
        num_times = int(len(self.original[:,0,0,0]))

        
        for t in range(num_times):
            tmp = data[t,0,:,:]
            psd, freq = rapsd(tmp, return_freq=True, normalize=True, fft_method=np.fft)            
            mean_spectral_density += psd
        mean_spectral_density /= num_times
        
        return mean_spectral_density, freq
    
    def run(self, num_times=None, timestamp=None):
        self.num_times = num_times
        self.timestamp = timestamp
        self.original_psd, self.freq_lr = self.compute_mean_spectral_density(self.original)
        self.generated_psd, self.freq_lr = self.compute_mean_spectral_density(self.generated)
        self.label_psd, self.freq = self.compute_mean_spectral_density(self.label)
        self.comp_psd, self.freq = self.compute_mean_spectral_density(self.comparison)
    
    
    def plot_psd(self, axis=None, fontsize=18, linewidth=None, model_resolution=0.25,
                 model_resolution_2=0.25, do_savefig=False, plt_legend=False):
        
        if axis is None: 
            #_, ax = plt.subplots(figsize=(7,6))
            _, ax = plt.subplots()
        else:
            ax = axis
        plt.rcParams.update({'font.size': 12})
        """
        resolution example: 1 degree / pixel  = ?km?
        equatorial circumference is approximately 40,075km
        40,075 kilometers / 360 degrees ≈ 111.32 km per degree
        -> 1 degree ≈ 111.32 km/degree
        
        for 0.25 res:
        1 pixel = 0.25 degrees/pixel
        1 degree ≈ 40,075 km / 360 degrees ≈ 111.32 km / degree
        0.25 degrees/pixel * 111.32 kilometers/degree ≈ 27.83
        """
        x_vals = 1/self.freq*model_resolution*111 /2 
        x_vals_lr = 1/self.freq_lr*model_resolution_2*111/2
        
        
        if self.plot_info_for_confused==True:
            print("len", len(self.freq), len(self.freq_lr))
            print("hr smallest km:",x_vals.min())
            print("lr smallest km:", x_vals_lr.min())

        
        ax.plot(x_vals_lr, self.original_psd, label= self.new_label_ori, linewidth=linewidth, color='#FFA500')   # gfdl       # orange  
        ax.plot(x_vals_lr, self.generated_psd , label= self.new_label_gen, linewidth=linewidth, color='#000000')  # era5       # black
        ax.plot(x_vals, self.label_psd , label= self.new_label_lab,  linewidth=linewidth, color='#0077BB')      # qm         # teal
        ax.plot(x_vals, self.comp_psd , label= self.new_label_comp, linewidth=linewidth, color='#FF00FF')       # DM         # magenta
        
        if plt_legend == True:
            ax.legend(loc='lower right', fontsize='small')
        #ax.legend()
        #ax.set_yscale('log', basey=2)
        #ax.set_xscale('log', basex=2)
        ax.set_yscale('log', base=10)
        ax.set_xscale('log', base=2)
        #ax.set_xticks([2**9, 2**10, 2**11, 2**12, 2**13])
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        #ax.grid()
        
        ax.set_xlabel(self.x_ax_name, fontsize=fontsize)
        ax.set_ylabel(self.y_ax_name, fontsize=fontsize)
        if self.title:
            ax.set_title(self.title,fontsize=fontsize)

        if do_savefig:
            plt.savefig(do_savefig)



class SpatialSpectralDensity_3_diff_res():
    """
    1 & 2 argument have a different resolution than argument 2&3 
    """
    def __init__(self, original, generated, label,
                 new_labels = ["original","generated", "UNet"]
                 ,y_ax_name='PSD [a.u]',x_ax_name='Wavelength[km]'
                 ,plot_info_for_confused=False, title=None):
        
        self.original = original
        self.generated = generated  
        self.label = label  
        self.new_label_ori = new_labels[0]
        self.new_label_gen = new_labels[1]
        self.new_label_lab = new_labels[2]
        self.x_ax_name = x_ax_name
        self.y_ax_name = y_ax_name
        self.plot_info_for_confused = plot_info_for_confused
        self.title = title
   
    def compute_mean_spectral_density(self, data):
        
        num_frequencies = np.max(((data.shape[3]),(data.shape[3])))/2
        mean_spectral_density = np.zeros(int(num_frequencies))
        num_times = int(len(self.original[:,0,0,0]))

        
        for t in range(num_times):
            tmp = data[t,0,:,:]
            psd, freq = rapsd(tmp, return_freq=True, normalize=True, fft_method=np.fft)            
            mean_spectral_density += psd
        mean_spectral_density /= num_times
        
        return mean_spectral_density, freq
    
    def run(self, num_times=None, timestamp=None):
        self.num_times = num_times
        self.timestamp = timestamp
        self.original_psd, self.freq_lr = self.compute_mean_spectral_density(self.original)
        self.generated_psd, self.freq_lr = self.compute_mean_spectral_density(self.generated)
        self.label_psd, self.freq = self.compute_mean_spectral_density(self.label)
    
    
    def plot_psd(self, axis=None, fontsize=18, linewidth=None, model_resolution=0.25,
                 model_resolution_2=0.25, do_savefig=False, plt_legend=False):
        
        if axis is None: 
            #_, ax = plt.subplots(figsize=(7,6))
            _, ax = plt.subplots()
        else:
            ax = axis
        plt.rcParams.update({'font.size': 12})
        x_vals = 1/self.freq*model_resolution*111 /2 
        x_vals_lr = 1/self.freq_lr*model_resolution_2*111/2
        
        
        if self.plot_info_for_confused==True:
            print("len", len(self.freq), len(self.freq_lr))
            print("hr smallest km:",x_vals.min())
            print("lr smallest km:", x_vals_lr.min())

        
        ax.plot(x_vals_lr, self.original_psd, label= self.new_label_ori, linewidth=linewidth, color='#000000')   # gfdl       # orange  
        ax.plot(x_vals_lr, self.generated_psd , label= self.new_label_gen, linewidth=linewidth, color='#0077BB')  # era5       # black
        ax.plot(x_vals, self.label_psd , label= self.new_label_lab,  linewidth=linewidth, color='#FF00FF')      # qm         # teal
        
        if plt_legend == True:
            ax.legend(loc='lower right', fontsize='small')
        #ax.legend()
        #ax.set_yscale('log', basey=2)
        #ax.set_xscale('log', basex=2)
        ax.set_yscale('log', base=10)
        ax.set_xscale('log', base=2)
        #ax.set_xticks([2**9, 2**10, 2**11, 2**12, 2**13])
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        #ax.grid()
        
        ax.set_xlabel(self.x_ax_name, fontsize=fontsize)
        ax.set_ylabel(self.y_ax_name, fontsize=fontsize)
        if self.title:
            ax.set_title(self.title,fontsize=fontsize)

        if do_savefig:
            plt.savefig(do_savefig, dpi=1000,bbox_inches='tight')



def latitudinal_mean_four_np(original, generated, label, comp,
                             label_name=['Original', 'Generated', 'Unet', 'Comparison'],
                             y_ax_name='Precipitation', x_ax_name='Longitude',
                             title_name="Latitudinal mean", do_savefig=False,ax=None):
    
    original = np.mean(original, axis=(0, 1, 2))
    generated = np.mean(generated, axis=(0, 1, 2))
    label = np.mean(label, axis=(0, 1, 2))
    comp = np.mean(comp, axis=(0, 1, 2))
    
    # Set custom range from -90 to -26
    latitudes = np.arange(-90, -26, 0.25)
    
    if ax==None:
        fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0], color='#FFA500')   # gfdl       # orange  
    ax.plot(latitudes, generated, label=label_name[1], color='#000000')  # era5       # black
    ax.plot(latitudes, label, label=label_name[2], color='#0077BB')      # qm         # teal
    ax.plot(latitudes, comp, label=label_name[3], color='#FF00FF')       # DM         # magenta
    ax.set_xlabel(x_ax_name, fontsize=18)
    ax.set_ylabel(y_ax_name, fontsize=18)
    ax.set_title(title_name, fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    if do_savefig:  
        plt.savefig(do_savefig)  
        print("saving to", do_savefig)


def longitudinal_mean_four_np(original, generated, label, comp,
                             label_name=['Original', 'Generated', 'Unet', 'Comparison'],
                             y_ax_name='Precipitation', x_ax_name='Latitude',
                             title_name="Longitudinal mean",do_savefig=False, ax=None):
    
    original = np.mean(original, axis=(0, 1, 3))
    generated = np.mean(generated, axis=(0, 1, 3))
    label = np.mean(label, axis=(0, 1, 3))
    comp = np.mean(comp, axis=(0, 1, 3))
    
    # Set custom range from -90 to -26
    latitudes = np.arange(-64, 0, 0.25)
    
    if ax==None:
        fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0], color='#FFA500')   # gfdl       # orange  
    ax.plot(latitudes, generated, label=label_name[1], color='#000000')  # era5       # black
    ax.plot(latitudes, label, label=label_name[2], color='#0077BB')      # qm         # teal
    ax.plot(latitudes, comp, label=label_name[3], color='#FF00FF')       # DM         # magenta
    ax.set_xlabel(x_ax_name, fontsize=18)
    ax.set_ylabel(y_ax_name, fontsize=18)
    ax.set_title(title_name, fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    if do_savefig:  
        plt.savefig(do_savefig)  
        print("saving to",do_savefig)
        
    #plt.show()


def plot_absolute_errors(bc_gfdl, raw_gfdl, era5, qm_gfdl
                         ,log=True,bin_start=0,bin_end=151,y_ax_name='Absolute error'
                         ,x_ax_name='Precipitation [mm/d]'
                         ,label_name=['ERA5 vs GFDL',
                                      'ERA5 vs QM corrected GFDL', 
                                      'ERA5 vs DM corrected GFDL']
                         ,title_name='Absolute error between histograms'
                         ,do_savefig=False,xlim=150,ax=None):
        
        if ax==None:
            _, ax = plt.subplots()
        bins =  np.arange(0, bin_end)
        target_bin_values = np.histogram(era5, bins=bins, density=True)[0]
        
        
        generated_bin_values = np.histogram(bc_gfdl.flatten(), bins=bins, density=True)[0]
        gen_differences = abs(target_bin_values - generated_bin_values)
        ax.plot(np.arange(0, len(gen_differences)),
                     gen_differences,
                     label=label_name[0],
                     linewidth=2, color='#FFA500')   # gfdl       # orange  
        
        label_bin_values = np.histogram(raw_gfdl.flatten(), bins=bins, density=True)[0]
        label_differences = abs(target_bin_values - label_bin_values)
        ax.plot(np.arange(0, len(label_differences)),
                     label_differences,
                     label=label_name[1],
                     linewidth=2, color='#0077BB')      # qm         # teal
        
        
        comp_bin_values = np.histogram(qm_gfdl.flatten(), bins=bins, density=True)[0]
        comp_differences = abs(target_bin_values - comp_bin_values)
        ax.plot(np.arange(0, len(comp_differences)),
                     comp_differences,
                     label=label_name[2],
                     linewidth=2, color='#FF00FF')       # DM         # magenta
        

        ax.set_xlabel(x_ax_name, fontsize=18)
        ax.set_ylabel(y_ax_name, fontsize=18)
        ax.set_xlim(0, xlim)
        ax.set_ylim(1e-8, 1e-1)

        xticks_positions = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])
        ax.set_xticks(xticks_positions)
        if log:
            ax.set_yscale('log')
        #plt.legend(fontsize='small')
        #ax.legend()
        #ax.grid()
        ax.set_title(title_name, fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        if do_savefig:
            ax.savefig(do_savefig)
            print("Saving to", do_savefig)
        
        #plt.show()


def histograms_four_np(original, generated, label, comp, log=True, xlim_end=None, alpha_1=1,
                        alpha_2=1, alpha_3=1, label_name=['Original', 'Generated', 'Unet', 'Comparison'],
                        bins=100, y_ax_name='Precipitation', x_ax_name='Longitude',
                        title_name="Latitudinal mean", do_savefig=False,ax=None):
    if ax==None:
        _, ax = plt.subplots()
    original = original.flatten()
    generated = generated.flatten()
    label = label.flatten()
    comp = comp.flatten()

    _ = ax.hist(original,
                 bins=bins,
                 histtype='step',
                 log=log,
                 density=True,
                 alpha=alpha_1,
                 linewidth=2,
                 label=label_name[0],
                 color='#FFA500')   # gfdl       # orange  

    _ = ax.hist(generated,
                 bins=bins,
                 histtype='step',
                 log=log,
                 density=True,
                 alpha=alpha_2,
                 linewidth=2,
                 label=label_name[1],
                  color='#000000')  # era5       # black

    _ = ax.hist(label,
                 bins=bins,
                 histtype='step',
                 log=log,
                 density=True,
                 alpha=alpha_3,
                 linewidth=2,
                 label=label_name[2],
                 color='#0077BB')      # qm         # teal
    
    _ = ax.hist(comp,
                 bins=bins,
                 histtype='step',
                 log=log,
                 density=True,
                 alpha=alpha_3,
                 linewidth=2,
                 label=label_name[3],
                 color='#FF00FF')       # DM         # magenta

    ax.set_xlabel(x_ax_name, fontsize=18)
    ax.set_ylabel(y_ax_name, fontsize=18)
    ax.set_title(title_name, fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    if xlim_end:
        ax.set_xlim(0, xlim_end)
    #ax.grid()
    #ax.legend()
    
    if do_savefig:  
        ax.savefig(do_savefig)  
        print("saving to", do_savefig)


def plot_images_no_lab(images,do_savefig=False ):
    #images = images.cpu().detach().numpy()
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu().detach().numpy())
    num_images = len(images)
    tick_positions = [(i + 0.5) * images[0].shape[-1] for i in range(num_images)]
    #plt.xticks(tick_positions)
    plt.xticks([])
    plt.yticks([])
    if do_savefig:  
        plt.savefig(do_savefig)  
        print("saving to",do_savefig)
        
    plt.show()


def plot_images_with_captions(images, titles, do_savefig=False):
    fig, axs = plt.subplots(len(images), figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    for i, ax in enumerate(axs):
        ax.imshow(torch.cat([img.cpu() for img in images[i]], dim=-1).permute(1, 2, 0).cpu().detach().numpy())
        ax.set_xticks([])
        ax.set_yticks([])

    #
    if do_savefig:  
        plt.savefig(do_savefig)  
        print("Saving to", do_savefig)
    
    plt.show()


# # paper plots era5 eval

def latitudinal_profile_three(original, generated, label, ax=None,
                             label_name=['Original', 'Generated', 'Comparison'],
                             y_ax_name="Mean precipitation [mm/d]", x_ax_name='Longitude',
                             title_name="Latitudinal mean",do_savefig=False):
    
    original = np.mean(original, axis=(0, 1, 2))
    generated = np.mean(generated, axis=(0, 1, 2))
    label = np.mean(label, axis=(0, 1, 2))
    
    # Set custom range from -90 to -26
    latitudes = np.arange(-90, -26, 0.25)
    
    if ax==None:
        fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0], color='#FFA500')   # era5 lr       # orange  
    ax.plot(latitudes, generated, label=label_name[1], color='#000000')  # era5 hr       # black
    ax.plot(latitudes, label, label=label_name[2], color='#FF00FF')      # DM            # magenta
    ax.set_xlabel(x_ax_name, fontsize=18)
    ax.set_ylabel(y_ax_name, fontsize=18)
    ax.set_title(title_name, fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    #ax.legend()
    
    if do_savefig:  
        plt.savefig(do_savefig)  
        print("saving to",do_savefig)


def longitudinal_profile_three(original, generated, label, ax=None,
                             label_name=['Original', 'Generated', 'Comparison'],
                             y_ax_name="Mean precipitation [mm/d]", x_ax_name='Latitude',
                             title_name="Longitudinal mean",do_savefig=False):
    
    original = np.mean(original, axis=(0, 1, 3))
    generated = np.mean(generated, axis=(0, 1, 3))
    label = np.mean(label, axis=(0, 1, 3))
    
    # Set custom range from -90 to -26
    latitudes = np.arange(-64, 0, 0.25)
    
    if ax==None:
        fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0], color='#FFA500')   # era5 lr       # orange  
    ax.plot(latitudes, generated, label=label_name[1], color='#000000')  # era5 hr       # black
    ax.plot(latitudes, label, label=label_name[2], color='#FF00FF')      # DM            # magenta
    ax.set_xlabel(x_ax_name, fontsize=18)
    ax.set_ylabel(y_ax_name, fontsize=18)
    ax.set_title(title_name, fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    if do_savefig:  
        plt.savefig(do_savefig)  
        print("saving to",do_savefig)
        
    #plt.show()  


def histograms_paper(original, generated, label, log=True, xlim_end=200, alpha_1=1,
                        alpha_2=1, alpha_3=1, label_name=['Original', 'Generated', 'Unet'],
                        bins=300, y_ax_name='Precipitation [mm/d]', x_ax_name='Frequency',
                        title_name="Histogram", do_savefig=False, ax=None):
    if ax==None:
        fig, ax = plt.subplots()
    #_, ax = plt.subplots()
    
    original = original.flatten()
    generated = generated.flatten()
    label = label.flatten()

    _ = ax.hist(original,
                 bins=bins,
                 histtype='step',
                 log=log,
                 density=True,
                 alpha=alpha_1,
                 linewidth=2,
                 label=label_name[0],
                 color='#FFA500')   # gfdl       # orange  

    _ = ax.hist(generated,
                 bins=bins,
                 histtype='step',
                 log=log,
                 density=True,
                 alpha=alpha_2,
                 linewidth=2,
                 label=label_name[1],
                  color='#000000')  # era5       # black

    _ = ax.hist(label,
                 bins=bins,
                 histtype='step',
                 log=log,
                 density=True,
                 alpha=alpha_3,
                 linewidth=2,
                 label=label_name[2],
                 color='#FF00FF')      
    

    ax.set_xlabel(x_ax_name, fontsize=18)
    ax.set_ylabel(y_ax_name, fontsize=18)
    ax.set_title(title_name, fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    if xlim_end:
        ax.set_xlim(0, xlim_end)
    
    if do_savefig:  
        ax.savefig(do_savefig)  
        print("saving to", do_savefig)
        
    #plt.show()


class SpatialSpectralDensity_diff_res():
    """
    1 argument has a different resolution than argument 2,3 
    """
    
    def __init__(self, original, generated, label, comparison=None, 
                 new_labels = ["original","generated", "UNet"],title=None):
        self.original = original
        self.generated = generated  
        self.label = label  
        self.new_label_ori = new_labels[0]
        self.new_label_gen = new_labels[1]
        self.new_label_lab = new_labels[2]
        self.title = title
    
        
    def compute_mean_spectral_density(self, data):
        num_frequencies = np.max(((data.shape[3]),(data.shape[3])))/2
        mean_spectral_density = np.zeros(int(num_frequencies))
        num_times = int(len(self.original[:,0,0,0]))

        
        for t in range(num_times):
            tmp = data[t,0,:,:]
            psd, freq = rapsd(tmp, return_freq=True, normalize=True, fft_method=np.fft)            
            mean_spectral_density += psd
        mean_spectral_density /= num_times
        
        return mean_spectral_density, freq
    
    def run(self, num_times=None, timestamp=None):
        self.num_times = num_times
        self.timestamp = timestamp
        self.original_psd, self.freq_lr = self.compute_mean_spectral_density(self.original)
        self.generated_psd, self.freq = self.compute_mean_spectral_density(self.generated)
        self.label_psd, self.freq = self.compute_mean_spectral_density(self.label)
    
    def plot_psd(self, axis=None, fname=None, fontsize=18, linewidth=None, model_resolution=0.5,
                 model_resolution_2=1, do_savefig=False, plt_legend=False):
        if axis is None: 
            _, ax = plt.subplots(figsize=(7,6))
        else:
            ax = axis
        plt.rcParams.update({'font.size': 12})
        
        """
        resolution example: 1 degree / pixel  = ?km?
        equatorial circumference is approximately 40,075km
        40,075 kilometers / 360 degrees ≈ 111.32 km per degree
        -> 1 degree ≈ 111.32 km/degree
        
        for 0.25 res:
        1 pixel = 0.25 degrees/pixel
        1 degree ≈ 40,075 km / 360 degrees ≈ 111.32 km / degree
        0.25 degrees/pixel * 111.32 kilometers/degree ≈ 27.83
        """
        x_vals = 1/self.freq*model_resolution*111 /2 #    why / 2 -> prbl. bc there is long, lat 
        x_vals_lr = 1/self.freq_lr*model_resolution_2*111/2
        
        
        #print("hr smallest km:",x_vals.min())
        #print("lr smallest km:", x_vals_lr.min())

        
        #x_vals = np.flip(1/self.freq*model_resolution*111/2)
        ax.plot(x_vals_lr, self.original_psd, label= self.new_label_ori,  linewidth=linewidth, color='#FFA500')   # era5 lr       # orange  
        ax.plot(x_vals, self.generated_psd , label= self.new_label_gen, linewidth=linewidth, color='#000000')  # era5 hr       # black
        ax.plot(x_vals, self.label_psd , label= self.new_label_lab, linewidth=linewidth, color='#FF00FF')      # DM            # magenta
        
        if plt_legend == True:
            ax.legend(loc='lower right', fontsize='small')
        #ax.legend(loc='lower right', fontsize=fontsize)
        
        #ax.set_yscale('log', basey=2)
        #ax.set_xscale('log', basex=2)
        ax.set_yscale('log', base=10)
        ax.set_xscale('log', base=2)
        #ax.set_xticks([2**9, 2**10, 2**11, 2**12, 2**13])
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        #ax.grid()
        
        
        ax.set_xlabel(r'Wavelength[km]', fontsize=fontsize)
        ax.set_ylabel('Power spectral density [a.u]', fontsize=fontsize)
        if self.title:
            ax.set_title("Spatial spectrum", fontsize=18)

        if do_savefig:
            plt.savefig(do_savefig)


# # single plots

def plot_absolute_errors_single(bc_gfdl, raw_gfdl, era5, qm_gfdl
                         ,log=True,bin_start=0,bin_end=151,y_ax_name='Absolute error'
                         ,x_ax_name='Precipitation [mm/d]'
                         ,label_name=['ERA5 vs GFDL',
                                      'ERA5 vs QM corrected GFDL', 
                                      'ERA5 vs DM corrected GFDL']
                         ,title_name='Absolute error between histograms'
                         ,do_savefig=False,xlim=150,ax=None):
        
        if ax==None:
            _, ax = plt.subplots()
        bins =  np.arange(0, bin_end)
        target_bin_values = np.histogram(era5, bins=bins, density=True)[0]
        
        
        generated_bin_values = np.histogram(bc_gfdl.flatten(), bins=bins, density=True)[0]
        gen_differences = abs(target_bin_values - generated_bin_values)
        ax.plot(np.arange(0, len(gen_differences)),
                     gen_differences,
                     label=label_name[0],
                     linewidth=2, color='#FFA500')   # gfdl       # orange  
        
        label_bin_values = np.histogram(raw_gfdl.flatten(), bins=bins, density=True)[0]
        label_differences = abs(target_bin_values - label_bin_values)
        ax.plot(np.arange(0, len(label_differences)),
                     label_differences,
                     label=label_name[1],
                     linewidth=2, color='#0077BB')      # qm         # teal
        
        
        comp_bin_values = np.histogram(qm_gfdl.flatten(), bins=bins, density=True)[0]
        comp_differences = abs(target_bin_values - comp_bin_values)
        ax.plot(np.arange(0, len(comp_differences)),
                     comp_differences,
                     label=label_name[2],
                     linewidth=2, color='#FF00FF')       # DM         # magenta
        

        ax.set_xlabel(x_ax_name)#, fontsize=18)
        ax.set_ylabel(y_ax_name)#, fontsize=18)
        ax.set_xlim(0, xlim)
        ax.set_ylim(1e-8, 1e-1)

        xticks_positions = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])
        ax.set_xticks(xticks_positions)
        if log:
            ax.set_yscale('log')
        ax.legend(fontsize='small')
        ax.legend()
        #ax.grid()
        ax.set_title(title_name)#, fontsize=18)
        #ax.tick_params(axis='x')#, labelsize=16)
        #ax.tick_params(axis='y', labelsize=16)

        if do_savefig:
            plt.savefig(do_savefig)
            print("Saving to", do_savefig)
        
        plt.show()


def longitudinal_mean_single(original, generated, label, comp,
                             label_name=['Original', 'Generated', 'Unet', 'Comparison'],
                             y_ax_name='Precipitation', x_ax_name='Latitude',
                             title_name="Longitudinal mean",do_savefig=False, ax=None):
    
    original = np.mean(original, axis=(0, 1, 3))
    generated = np.mean(generated, axis=(0, 1, 3))
    label = np.mean(label, axis=(0, 1, 3))
    comp = np.mean(comp, axis=(0, 1, 3))
    
    # Set custom range from -90 to -26
    latitudes = np.arange(-64, 0, 0.25)
    
    if ax==None:
        fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0], color='#FFA500')   # gfdl       # orange  
    ax.plot(latitudes, generated, label=label_name[1], color='#000000')  # era5       # black
    ax.plot(latitudes, label, label=label_name[2], color='#0077BB')      # qm         # teal
    ax.plot(latitudes, comp, label=label_name[3], color='#FF00FF')       # DM         # magenta
    ax.set_xlabel(x_ax_name)#, fontsize=16)
    ax.set_ylabel(y_ax_name)#, fontsize=16)
    ax.set_title(title_name)#, fontsize=16)
    #ax.tick_params(axis='x', labelsize=12)
    #ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize="small")
    if do_savefig:  
        plt.savefig(do_savefig)  
        print("saving to",do_savefig)
        
    plt.show()



def latitudinal_mean_3_paper(original, generated, label, 
                             label_name=['Original', 'Generated', 'Unet'],
                             y_ax_name='Precipitation', x_ax_name='Longitude',
                             title_name="Latitudinal mean", do_savefig=False,ax=None):
    
    original = np.mean(original, axis=(0, 1, 2))
    generated = np.mean(generated, axis=(0, 1, 2))
    label = np.mean(label, axis=(0, 1, 2))
    
    # Set custom range from -90 to -26
    latitudes = np.arange(-90, -26, 0.25)
    
    if ax==None:
        fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0], color='#FFA500')   # gfdl       # orange  
    ax.plot(latitudes, generated, label=label_name[1], color='#000000')  # era5       # black
    ax.plot(latitudes, label, label=label_name[2], color='#0077BB')      # qm         # teal
    ax.set_xlabel(x_ax_name, fontsize=18)
    ax.set_ylabel(y_ax_name, fontsize=18)
    ax.set_title(title_name, fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    if do_savefig:  
        plt.savefig(do_savefig)  
        print("saving to", do_savefig)

def longitudinal_mean_3_paper(original, generated, label, 
                             label_name=['Original', 'Generated', 'Unet'],
                             y_ax_name='Precipitation', x_ax_name='Latitude',
                             title_name="Longitudinal mean",do_savefig=False, ax=None):
    
    original = np.mean(original, axis=(0, 1, 3))
    generated = np.mean(generated, axis=(0, 1, 3))
    label = np.mean(label, axis=(0, 1, 3))
    
    # Set custom range from -90 to -26
    latitudes = np.arange(-64, 0, 0.25)
    
    if ax==None:
        fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0], color='#FFA500')   # gfdl       # orange  
    ax.plot(latitudes, generated, label=label_name[1], color='#000000')  # era5       # black
    ax.plot(latitudes, label, label=label_name[2], color='#0077BB')      # qm         # teal
    ax.set_xlabel(x_ax_name, fontsize=18)
    ax.set_ylabel(y_ax_name, fontsize=18)
    ax.set_title(title_name, fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    if do_savefig:  
        plt.savefig(do_savefig)  
        print("saving to",do_savefig)




def histograms_3_paper(original, generated, label, log=True, xlim_end=None, alpha_1=1,
                        alpha_2=1, alpha_3=1, label_name=['Original', 'Generated', 'Unet'],
                        bins=100, y_ax_name='Precipitation', x_ax_name='Longitude',
                        title_name="Latitudinal mean", do_savefig=False,ax=None):
    if ax==None:
        _, ax = plt.subplots()
    original = original.flatten()
    generated = generated.flatten()
    label = label.flatten()

    _ = ax.hist(original,
                 bins=bins,
                 histtype='step',
                 log=log,
                 density=True,
                 alpha=alpha_1,
                 linewidth=2,
                 label=label_name[0],
                 color='#FFA500')   # gfdl       # orange  

    _ = ax.hist(generated,
                 bins=bins,
                 histtype='step',
                 log=log,
                 density=True,
                 alpha=alpha_2,
                 linewidth=2,
                 label=label_name[1],
                  color='#000000')  # era5       # black

    _ = ax.hist(label,
                 bins=bins,
                 histtype='step',
                 log=log,
                 density=True,
                 alpha=alpha_3,
                 linewidth=2,
                 label=label_name[2],
                 color='#0077BB')      # qm         # teal


    ax.set_xlabel(x_ax_name, fontsize=18)
    ax.set_ylabel(y_ax_name, fontsize=18)
    ax.set_title(title_name, fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    if xlim_end:
        ax.set_xlim(0, xlim_end)
    #ax.grid()
    #ax.legend()
    
    if do_savefig:  
        ax.savefig(do_savefig)  
        print("saving to", do_savefig)



class SpatialSpectralDensity_diff_3_paper():
    """
    1 & 2 argument have a different resolution than argument 2&3 
    """
    def __init__(self, original, generated, label, 
                 new_labels = ["original","generated", "UNet"]
                 ,y_ax_name='PSD [a.u]',x_ax_name='Wavelength[km]'
                 ,plot_info_for_confused=False, title=None):
        
        self.original = original
        self.generated = generated  
        self.label = label  
        self.new_label_ori = new_labels[0]
        self.new_label_gen = new_labels[1]
        self.new_label_lab = new_labels[2]
        self.x_ax_name = x_ax_name
        self.y_ax_name = y_ax_name
        self.plot_info_for_confused = plot_info_for_confused
        self.title = title
   
    def compute_mean_spectral_density(self, data):
        
        num_frequencies = np.max(((data.shape[3]),(data.shape[3])))/2
        mean_spectral_density = np.zeros(int(num_frequencies))
        num_times = int(len(self.original[:,0,0,0]))

        
        for t in range(num_times):
            tmp = data[t,0,:,:]
            psd, freq = rapsd(tmp, return_freq=True, normalize=True, fft_method=np.fft)            
            mean_spectral_density += psd
        mean_spectral_density /= num_times
        
        return mean_spectral_density, freq
    
    def run(self, num_times=None, timestamp=None):
        self.num_times = num_times
        self.timestamp = timestamp
        self.original_psd, self.freq_lr = self.compute_mean_spectral_density(self.original)
        self.generated_psd, self.freq_lr = self.compute_mean_spectral_density(self.generated)
        self.label_psd, self.freq = self.compute_mean_spectral_density(self.label)
    
    
    def plot_psd(self, axis=None, fontsize=18, linewidth=None, model_resolution=0.25,
                 model_resolution_2=0.25, do_savefig=False, plt_legend=False):
        
        if axis is None: 
            #_, ax = plt.subplots(figsize=(7,6))
            _, ax = plt.subplots()
        else:
            ax = axis
        plt.rcParams.update({'font.size': 12})
        """
        resolution example: 1 degree / pixel  = ?km?
        equatorial circumference is approximately 40,075km
        40,075 kilometers / 360 degrees ≈ 111.32 km per degree
        -> 1 degree ≈ 111.32 km/degree
        
        for 0.25 res:
        1 pixel = 0.25 degrees/pixel
        1 degree ≈ 40,075 km / 360 degrees ≈ 111.32 km / degree
        0.25 degrees/pixel * 111.32 kilometers/degree ≈ 27.83
        """
        x_vals = 1/self.freq*model_resolution*111 /2 
        x_vals_lr = 1/self.freq_lr*model_resolution_2*111/2
        
        
        if self.plot_info_for_confused==True:
            print("len", len(self.freq), len(self.freq_lr))
            print("hr smallest km:",x_vals.min())
            print("lr smallest km:", x_vals_lr.min())

        
        ax.plot(x_vals_lr, self.original_psd, label= self.new_label_ori, linewidth=linewidth, color='#FFA500')   # gfdl       # orange  
        ax.plot(x_vals_lr, self.generated_psd , label= self.new_label_gen, linewidth=linewidth, color='#000000')  # era5       # black
        ax.plot(x_vals, self.label_psd , label= self.new_label_lab,  linewidth=linewidth, color='#0077BB')      # qm         # teal
        
        if plt_legend == True:
            ax.legend(loc='lower right', fontsize='small')
        #ax.legend()
        #ax.set_yscale('log', basey=2)
        #ax.set_xscale('log', basex=2)
        ax.set_yscale('log', base=10)
        ax.set_xscale('log', base=2)
        #ax.set_xticks([2**9, 2**10, 2**11, 2**12, 2**13])
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        #ax.grid()
        
        ax.set_xlabel(self.x_ax_name, fontsize=fontsize)
        ax.set_ylabel(self.y_ax_name, fontsize=fontsize)
        if self.title:
            ax.set_title(self.title,fontsize=fontsize)

        if do_savefig:
            plt.savefig(do_savefig)