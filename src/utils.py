import matplotlib.pyplot as plt
import numpy as np
import torch


def histograms_three_np(original, generated, label, log=True, xlim_end=None, var="p",alpha_1=1,density=True,
                        alpha_2=1, alpha_3=1,bins=100,label_name=['Original', 'Generated', 'Unet'],title=None
                       , y_ax_name="Frequency", x_ax_name="Precipitation  [mm/d]", do_savefig=None):
    _, ax = plt.subplots()

    original = original.flatten()
    generated = generated.flatten()
    label = label.flatten()

    _ = plt.hist(original,
                bins=bins,
                histtype='step',
                log=log,
                density=density,
                alpha=alpha_1,
                linewidth=2,
                label=label_name[0])


    _ = plt.hist(generated,
                bins=bins,
                histtype='step',
                log=log,
                density=density,
                alpha=alpha_2,
                linewidth=2,
                label=label_name[1])
    
    
    _ = plt.hist(label,
                bins=bins,
                histtype='step',
                log=log,
                density=density,
                alpha=alpha_3,
                linewidth=2,
                label=label_name[2])
    
    
    plt.xlabel(x_ax_name)
    plt.ylabel(y_ax_name)
    if xlim_end:
        plt.xlim(0, xlim_end)
    
    if title:
        plt.title(title)
    plt.grid()
    plt.legend()
    
    if do_savefig:  
        plt.savefig(do_savefig)  
        print("saving to",do_savefig)
    plt.show()


def latitudinal_mean_four_np(original, generated, label, comp,
                             label_name=['Original', 'Generated', 'Unet', 'Comparison'],
                             y_ax_name='Precipitation', x_ax_name='Longitude',
                             title_name="Latitudinal mean plot", do_savefig=False, legend=False):
    
    original = np.mean(original, axis=(0, 1, 2))
    generated = np.mean(generated, axis=(0, 1, 2))
    label = np.mean(label, axis=(0, 1, 2))
    comp = np.mean(comp, axis=(0, 1, 2))
    
    # Set custom range from -90 to -26
    latitudes = np.arange(-90, -26, 0.25)
    
    fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0], color='#FFA500')   # gfdl       # orange  
    ax.plot(latitudes, generated, label=label_name[1], color='#000000')  # era5       # black
    ax.plot(latitudes, label, label=label_name[2], color='#0077BB')      # qm         # teal
    ax.plot(latitudes, comp, label=label_name[3], color='#FF00FF')       # DM         # magenta
    ax.set_xlabel(x_ax_name)
    ax.set_ylabel(y_ax_name)
    ax.legend()
    if legend==True:
        ax.legend(fontsize='small')

    ax.set_title(title_name)
    
    if do_savefig:  
        plt.savefig(do_savefig)  
        print("saving to", do_savefig)
        
    plt.show()


def plot_absolute_errors(bc_gfdl, raw_gfdl, era5, qm_gfdl
                         ,log=True,bin_start=0,bin_end=151,y_ax_name='Absolute error'
                         ,x_ax_name='Precipitation [mm/d]'
                         ,label_name=['ERA5 vs GFDL', 'ERA5 vs QM-corrected GFDL', 
                                  'ERA5 vs DM-corrected GFDL']
                         ,title_name='Absolute error between histograms'
                         ,do_savefig=False,xlim=150):

        bins =  np.arange(0, bin_end)
        target_bin_values = np.histogram(era5, bins=bins, density=True)[0]
        
        
        generated_bin_values = np.histogram(bc_gfdl.flatten(), bins=bins, density=True)[0]
        gen_differences = abs(target_bin_values - generated_bin_values)
        plt.plot(np.arange(0, len(gen_differences)),
                     gen_differences,
                     label=label_name[0],
                     linewidth=2, color='#FFA500')   # gfdl       # orange  
        
        label_bin_values = np.histogram(raw_gfdl.flatten(), bins=bins, density=True)[0]
        label_differences = abs(target_bin_values - label_bin_values)
        plt.plot(np.arange(0, len(label_differences)),
                     label_differences,
                     label=label_name[1],
                     linewidth=2, color='#0077BB')      # qm         # teal
        
        
        comp_bin_values = np.histogram(qm_gfdl.flatten(), bins=bins, density=True)[0]
        comp_differences = abs(target_bin_values - comp_bin_values)
        plt.plot(np.arange(0, len(comp_differences)),
                     comp_differences,
                     label=label_name[2],
                     linewidth=2, color='#FF00FF')       # DM         # magenta
        

        plt.ylabel(y_ax_name)
        plt.xlabel(x_ax_name)
        plt.xlim(0, xlim)
        plt.ylim(1e-8, 1e-1)

        xticks_positions = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])
        plt.xticks(xticks_positions)
        if log:
            plt.yscale('log')
        #plt.legend(fontsize='small')
        plt.legend()
        plt.grid()
        plt.title(title_name)
        
        if do_savefig:
            plt.savefig(do_savefig)
            print("Saving to", do_savefig)
        
        plt.show()


def histograms_four_np(original, generated, label, comp, log=True, xlim_end=None, alpha_1=1,
                        alpha_2=1, alpha_3=1, label_name=['Original', 'Generated', 'Unet', 'Comparison'],
                        bins=100, y_ax_name='Precipitation', x_ax_name='Longitude',
                        title_name="Latitudinal mean plot", do_savefig=False):
    
    _, ax = plt.subplots()
    original = original.flatten()
    generated = generated.flatten()
    label = label.flatten()
    comp = comp.flatten()

    _ = plt.hist(original,
                 bins=bins,
                 histtype='step',
                 log=log,
                 density=True,
                 alpha=alpha_1,
                 linewidth=2,
                 label=label_name[0],
                 color='#FFA500')   # gfdl       # orange  

    _ = plt.hist(generated,
                 bins=bins,
                 histtype='step',
                 log=log,
                 density=True,
                 alpha=alpha_2,
                 linewidth=2,
                 label=label_name[1],
                  color='#000000')  # era5       # black

    _ = plt.hist(label,
                 bins=bins,
                 histtype='step',
                 log=log,
                 density=True,
                 alpha=alpha_3,
                 linewidth=2,
                 label=label_name[2],
                 color='#0077BB')      # qm         # teal
    
    _ = plt.hist(comp,
                 bins=bins,
                 histtype='step',
                 log=log,
                 density=True,
                 alpha=alpha_3,
                 linewidth=2,
                 label=label_name[3],
                 color='#FF00FF')       # DM         # magenta

    plt.xlabel(y_ax_name)
    plt.ylabel(x_ax_name)
    if xlim_end:
        plt.xlim(0, xlim_end)
    plt.grid()
    plt.legend()
    #plt.legend(fontsize='small')
    plt.title(title_name)
    
    if do_savefig:  
        plt.savefig(do_savefig)  
        print("saving to", do_savefig)
        
    plt.show()


def time_mean_three_np(original, generated, label, var="p", label_name=['era5', 'gfdl', 'dm gfdl']): 
    original = original.mean(axis=(1, 2, 3))
    generated = generated.mean(axis=(1, 2, 3))
    label = label.mean(axis=(1, 2, 3))
    
    latitudes = range(original.shape[0])  
    
    fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0])
    ax.plot(latitudes, generated, label=label_name[1])
    ax.plot(latitudes, label, label=label_name[2])
    ax.set_xlabel('Days')
    if var=="t":
        ax.set_ylabel('Temperature')
    else:
        ax.set_ylabel('Precipitation')
    ax.legend()
    plt.show()


def time_mean_three(original, generated, label, var="p", label_name=['era5', 'gfdl', 'dm gfdl']): 
    original = original.cpu().mean(dim=(1, 2, 3))
    generated = generated.cpu().mean(dim=(1, 2, 3))
    label = label.cpu().mean(dim=(1, 2, 3))
    
    latitudes = range(original.shape[0])  
    
    fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0])
    ax.plot(latitudes, generated, label=label_name[1])
    ax.plot(latitudes, label, label=label_name[2])
    ax.set_xlabel('Days')
    if var=="t":
        ax.set_ylabel('Temperature')
    else:
        ax.set_ylabel('Precipitation')
    ax.legend()
    plt.show()


def histograms(original, generated, log=True, xlim_end=None, var="p"):
    _, ax = plt.subplots()

    original = original.cpu().flatten()
    generated = generated.cpu().flatten()

    _ = plt.hist(original,
                bins=100,
                histtype='step',
                log=log,
                density=True,
                linewidth=2,
                label='Data')


    _ = plt.hist(generated,
                bins=100,
                histtype='step',
                log=log,
                density=True,
                linewidth=2,
                label='Generated')
    
    if var=="t":
        plt.xlabel('Mean Temperature [K]')
    else:
        plt.xlabel('Mean Precipitation [mm/d]')
    plt.ylabel('Histogram')
    if xlim_end:
        plt.xlim(0, xlim_end)
    plt.grid()
    plt.legend()
    plt.show()


def histograms_three(original, generated, label, log=True, xlim_end=None, var="p", label_name=['Original', 'Generated', 'Unet']):
    _, ax = plt.subplots()

    original = original.cpu().flatten()
    generated = generated.cpu().flatten()
    label = label.cpu().flatten()

    _ = plt.hist(original,
                bins=100,
                histtype='step',
                log=log,
                density=True,
                linewidth=2,
                label=label_name[0])


    _ = plt.hist(generated,
                bins=100,
                histtype='step',
                log=log,
                density=True,
                linewidth=2,
                label=label_name[1])
    
    
    _ = plt.hist(label,
                bins=100,
                histtype='step',
                log=log,
                density=True,
                linewidth=2,
                label=label_name[2])
    
    
    if var=="t":
        plt.xlabel('Mean Temperature [K]')
    else:
        plt.xlabel('Mean Precipitation [mm/d]')
    plt.ylabel('Histogram')
    if xlim_end:
        plt.xlim(0, xlim_end)
    plt.grid()
    plt.legend()
    plt.show()


def latitudinal_mean(original, generated, var="p"): 
    original = original.cpu().mean(dim=(0, 1, 2))
    generated = generated.cpu().mean(dim=(0, 1, 2))
    
    latitudes = range(original.shape[0])  
    
    fig, ax = plt.subplots()
    ax.plot(latitudes, original, label='original')
    ax.plot(latitudes, generated, label='generated')
    ax.set_xlabel('Longitude')
    if var=="t":
        ax.set_ylabel('Temperature')
    else:
        ax.set_ylabel('Precipitation')
    ax.legend()
    plt.show()


def latitudinal_mean_three(original, generated, label, var="p", label_name=['Original', 'Generated', 'Unet']): 
    original = original.cpu().mean(dim=(0, 1, 2))
    generated = generated.cpu().mean(dim=(0, 1, 2))
    label = label.cpu().mean(dim=(0, 1, 2))
    
    latitudes = range(original.shape[0])  
    
    fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0])
    ax.plot(latitudes, generated, label=label_name[1])
    ax.plot(latitudes, label, label=label_name[2])
    ax.set_xlabel('Longitude')
    if var=="t":
        ax.set_ylabel('Temperature')
    else:
        ax.set_ylabel('Precipitation')
    ax.legend()
    plt.show()


def latitudinal_mean_three_np_256(original, generated, label, var="p",
                              label_name=['Original', 'Generated', 'Unet'], do_savefig=None,
                              y_ax_name='Precipitation', x_ax_name='Longitude',
                              title_name="Latitudinal mean plot",): 
    
    original = np.mean(original, axis=(0, 1, 2))
    generated = np.mean(generated, axis=(0, 1, 2))
    label = np.mean(label, axis=(0, 1, 2))
    
    latitudes = np.arange(-90, -26, 0.25)
    
    fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0])
    ax.plot(latitudes, generated, label=label_name[1])
    ax.plot(latitudes, label, label=label_name[2])
    ax.set_xlabel(x_ax_name)
    ax.set_ylabel(y_ax_name)
    ax.legend()
    ax.set_title(title_name)
    
    if do_savefig:  
        plt.savefig(do_savefig)  
        print("saving to",do_savefig)
    plt.show()


def latitudinal_mean_three_np(original, generated, label, var="p",
                              label_name=['Original', 'Generated', 'Unet'], do_savefig=None,
                              y_ax_name='Precipitation', x_ax_name='Longitude',
                              title_name="Latitudinal mean plot",): 
    
    original = np.mean(original, axis=(0, 1, 2))
    generated = np.mean(generated, axis=(0, 1, 2))
    label = np.mean(label, axis=(0, 1, 2))
    
    latitudes = np.arange(-90, -26, 1)
    
    fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0])
    ax.plot(latitudes, generated, label=label_name[1])
    ax.plot(latitudes, label, label=label_name[2])
    ax.set_xlabel(x_ax_name)
    ax.set_ylabel(y_ax_name)
    ax.legend()
    ax.set_title(title_name)
    
    if do_savefig:  
        plt.savefig(do_savefig)  
        print("saving to",do_savefig)
    plt.show()


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


def longitudinal_mean_three(original, generated, label, var="p", label_name=['Original', 'Generated', 'Unet']): 
    original = original.cpu().mean(dim=(0, 1, 3))
    generated = generated.cpu().mean(dim=(0, 1, 3))
    label = label.cpu().mean(dim=(0, 1, 3))
    
    latitudes = range(original.shape[0])  
    
    fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0])
    ax.plot(latitudes, generated, label=label_name[1])
    ax.plot(latitudes, label, label=label_name[2])
    ax.set_xlabel('Latitude')
    if var=="t":
        ax.set_ylabel('Temperature')
    else:
        ax.set_ylabel('Precipitation')
    ax.legend()
    plt.show()


def longitudinal_mean_three_np(original, generated, label, var="p",
                              label_name=['Original', 'Generated', 'Unet'], do_savefig=None,
                              y_ax_name='Precipitation', x_ax_name='Longitude',
                              title_name="Latitudinal mean plot",): 
    
    original = np.mean(original, axis=(0, 1, 3))
    generated = np.mean(generated, axis=(0, 1, 3))
    label = np.mean(label, axis=(0, 1, 3))
    
    latitudes = np.arange(-64, 0, 1)
    
    fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0])
    ax.plot(latitudes, generated, label=label_name[1])
    ax.plot(latitudes, label, label=label_name[2])
    ax.set_xlabel(x_ax_name)
    ax.set_ylabel(y_ax_name)
    ax.legend()
    ax.set_title(title_name)
    
    if do_savefig:  
        plt.savefig(do_savefig)  
        print("saving to",do_savefig)
    plt.show()


def longitudinal_mean_four_np(original, generated, label, comp,
                             label_name=['Original', 'Generated', 'Unet', 'Comparison'],
                             y_ax_name='Precipitation', x_ax_name='Latitude',
                             title_name="Longitudinal mean plot",do_savefig=False):
    
    original = np.mean(original, axis=(0, 1, 3))
    generated = np.mean(generated, axis=(0, 1, 3))
    label = np.mean(label, axis=(0, 1, 3))
    comp = np.mean(comp, axis=(0, 1, 3))
    
    # Set custom range from -90 to -26
    latitudes = np.arange(-64, 0, 0.25)
    
    fig, ax = plt.subplots()
    ax.plot(latitudes, original, label=label_name[0], color='#FFA500')   # gfdl       # orange  
    ax.plot(latitudes, generated, label=label_name[1], color='#000000')  # era5       # black
    ax.plot(latitudes, label, label=label_name[2], color='#0077BB')      # qm         # teal
    ax.plot(latitudes, comp, label=label_name[3], color='#FF00FF')       # DM         # magenta
    ax.set_xlabel(x_ax_name)
    ax.set_ylabel(y_ax_name)
    #ax.legend(fontsize='small')
    ax.legend()
    #ax.set_title(title_name)
    
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
                             title_name="Latitudinal mean plot",do_savefig=False):
    
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
    ax.set_xlabel(x_ax_name)
    ax.set_ylabel(y_ax_name)
    #ax.legend()
    ax.set_title(title_name)
    
    if do_savefig:  
        plt.savefig(do_savefig)  
        print("saving to",do_savefig)
        
    #plt.show() 


def longitudinal_profile_three(original, generated, label, ax=None,
                             label_name=['Original', 'Generated', 'Comparison'],
                             y_ax_name="Mean precipitation [mm/d]", x_ax_name='Latitude',
                             title_name="Longitudinal mean plot",do_savefig=False):
    
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
    ax.set_xlabel(x_ax_name)
    ax.set_ylabel(y_ax_name)
    #ax.legend()
    ax.set_title(title_name)
    
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
    

    ax.set_xlabel(y_ax_name)
    ax.set_ylabel(x_ax_name)
    if xlim_end:
        ax.set_xlim(0, xlim_end)
    #ax.grid()
    #ax.legend()
    ax.set_title(title_name)
    
    if do_savefig:  
        ax.savefig(do_savefig)  
        print("saving to", do_savefig)
        
    #plt.show()
