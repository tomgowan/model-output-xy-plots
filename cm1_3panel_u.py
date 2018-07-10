import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pygrib, os, sys
from netCDF4 import Dataset
from numpy import *
import numpy as np
from pylab import *
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import nclcmaps
import pandas as pd
import xarray as xr

## Set varuable


#%%

### Read in cm1 file using xarray

#weak wind
ds_weak = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/old_runs/cm1out_200m_weak_wind_wide.nc')

#mod wind
ds_mod = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/old_runs/cm1out_200m_normal_wind_wide.nc')

#string wind
ds_strong = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/old_runs/cm1out_200m_strong_wind_wide.nc')







#%%

##################### Create colormap from NCL library ########################

def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap


###############################################################################
############## Set ncl_cmap as the colormap you want ##########################
###############################################################################

### 1) In order for this to work the files "nclcmaps.py" and "__init__.py"
### must be present in the dirctory.
### 2) You must "import nclcmaps"
### 3) The path to nclcmaps.py must be added to tools -> PYTHONPATH manager in SPyder
### 4) Then click "Update module names list" in tolls in Spyder and restart Spyder
                
## The steps above describe the general steps for adding "non-built in" modules to Spyder

###############################################################################
###############################################################################


#Read in colormap and put in proper format
colors1 = np.array(nclcmaps.colors['precip_11lev'])#perc2_9lev'])
colors_int = colors1.astype(int)
colors = list(colors_int)
cmap_precip = make_cmap(colors, bit=True)

colors1_t = np.array(nclcmaps.colors['OceanLakeLandSnow'])
colors_int_t = colors1_t.astype(int)
colors_t = list(colors_int_t)
cmap_terrain = make_cmap(colors_t, bit=True)


#%%




    #############   Plots ########################################
    
    
for i in range(300,301):#35,len(ds_weak.cref[:,0,0,0])):

    secs = i*80
    
    fig = plt.figure(num=None, figsize=(12,11.6), dpi=800, facecolor='w', edgecolor='k')
    for j in range(1,4):
        subplot = 310 + j
        
        #Lbel to loop over runs
        run = ['ds_weak', 'ds_mod', 'ds_strong']
        model_run = eval(run[j-1])
        
        #Plot characteristics
        ax = plt.subplot(subplot,aspect = 'equal')
        plt.subplots_adjust(left=0.04, bottom=0.01, right=0.9, top=0.95, wspace=0, hspace=0)
        plt.axis('equal')
        plt.axis('off')
        
        
        #Plot variable details
        levels_cref = np.arange(10,35.01,0.1)
        levels_ticks = np.arange(10,35.02,5)
        cref_plot = plt.contourf(model_run.cref[i,:,:], levels_cref, cmap = cm.YlGnBu, alpha = 1, extend = 'max', zorder = 3) #vmin = -10,
    
        #Water and land details
        levels_water = [1.5, 2.5]
        levels_terrain = [0,1.5]
        xland_plt = plt.contourf(ds_weak.xland[0,:,:], levels_water, alpha = 1, colors = ('lightsteelblue'), zorder = 2)
        xland_plt = plt.contourf(ds_weak.xland[0,:,:], levels_terrain, alpha = 1, colors = ('gainsboro'), zorder = 1)
        
        #Labels
        sub_title = ['Weak Wind', 'Moderate Wind', 'Strong Wind']
        props = dict(boxstyle='square', facecolor='white', alpha=1)
        ax.text(20, 350, sub_title[j-1], fontsize = 17, bbox = props, zorder = 5)
        
        if j == 1:
            #Title
            plt.title("Composite Reflectivity [dBZ] (elapsed time = %d seconds)"  % secs, fontsize = 22, y = 0.98) 

    
    
    
    #Colorbar
    cbaxes = fig.add_axes([0.93, 0.2, 0.035, 0.55])             
    cbar = plt.colorbar(cref_plot, cax = cbaxes, ticks = levels_ticks)
    cbar.ax.tick_params(labelsize=15)
    
    
    plt.savefig("/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/varied_wind_wide_cref%03d.png" % i)

    plt.close(fig)



