import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.basemap
from mpl_toolkits.basemap import Basemap, maskoceans
import pygrib, os, sys
from netCDF4 import Dataset
from numpy import *
import numpy as np
from pylab import *
import time
from datetime import date, timedelta
from matplotlib import animation
import matplotlib.animation as animation
import types
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import nclcmaps
import pandas as pd
import xarray as xr

## Set varuable



### Read in with netcdf
#cm1_file = "/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/cm1out_lake_200m_skinny_lake.nc"
#fh = Dataset(cm1_file)
#
#cref = fh.variables['cref'][:,:,:]
#xland  = fh.variables['xland'][:]
##zs  = fh.variables['zs'][:]


#Read in with xarray
ds_no = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/cm1out_200m_20ms_1500m.nc')
ds_small = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/cm1out_200m_20ms_1500m.nc')
ds_tall = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/cm1out_200m_20ms_2500m.nc')

#%%




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
colors1 = np.array(nclcmaps.colors['MPL_YlGnBu'])#perc2_9lev'])
colors_int = colors1.astype(int)
colors = list(colors_int)
cmap_var = nclcmaps.make_cmap(colors, bit=True)

colors1_t = np.array(nclcmaps.colors['MPL_Greys'])
colors_int_t = colors1_t.astype(int)
colors_t = list(colors_int_t)
cmap_terrain = nclcmaps.make_cmap(colors_t, bit=True)


#%%



#%%
##############################   Plots ########################################
    
    
#for i in range(130,131):
for i in range(0,np.min([ds_no.dbz[:,0,0,0].size, ds_small.dbz[:,0,0,0].size, ds_tall.dbz[:,0,0,0].size])):
    
    secs = i*120

    
    fig = plt.figure(num=None, figsize=(12,11.6), facecolor='w', edgecolor='k')
    for j in range(1,4):
        subplot = 310 + j
        
        #Lbel to loop over runs
        run = ['ds_no', 'ds_small', 'ds_tall']
        model_run = eval(run[j-1])
        
        #Plot characteristics
        ax = plt.subplot(subplot,aspect = 'equal')
        plt.subplots_adjust(left=0.04, bottom=0.01, right=0.88, top=0.95, wspace=0, hspace=0)
        plt.axis('equal')
        plt.axis('off')
        
        
        #Create grid
        x = np.arange(0,1300,1)
        y = np.arange(0,400,1)
        
        #Levels for theta
        levels = np.arange(257,266.001,.1)
        levels_ticks = np.arange(257,266.001,1)
        
        #Plot theta
        cref_plot = plt.contourf(model_run.th[i,0,:,:], levels, cmap = cm.jet, extend = 'both', alpha = 1,  zorder = 3)
    
        #Plot land, water, terrain
        levels_water = [1.5, 2.5]
        levels_terrain = [0,1.5]
        terrain_levels = np.arange(-1, 3000.1, 200)
        terrain_ticks = np.arange(0,3000.1,500)
        
        water = plt.contourf(model_run.xland[0,:,:], levels_water, alpha = 1, colors = ('lightsteelblue'), zorder = 2)
        land = plt.contourf(model_run.xland[0,:,:], levels_terrain, alpha = 1, colors = ('gainsboro'), zorder = 1)
        try:
            terrain = plt.contourf(model_run.zs[0,:,:], terrain_levels, alpha = 1, cmap = cm.Greys, vmin = -800, vmax = 3000, zorder = 1)
        except:
            pass
        
        # Calculate Wind Barbs
        yy = np.arange(40, len(model_run.u[i,0,:,0]), 40)
        xx = np.arange(40, len(model_run.u[i,0,0,:]), 40)
        points = np.meshgrid(yy, xx)
        x, y = np.meshgrid(yy, xx)
        U = np.array(model_run.u[i,0,:,:])
        V = np.array(model_run.v[i,0,:,:])
        barbs = ax.barbs(y, x, U[points], V[points], length = 5, zorder = 4, barbcolor = 'w')
        
        #Labels
        sub_title = ['1500m Mountain/20ms', '1500m Mountain/20ms', '2500m Mountain/20ms']
        props = dict(boxstyle='square', facecolor='white', alpha=1)
        ax.text(20, 450, sub_title[j-1], fontsize = 17, bbox = props, zorder = 5)
        
        if j == 1:
            #Title
            plt.title("Surface Theta and Wind [elapsed time = %d seconds]"  % secs, fontsize = 22, y = 0.98) 
    
    #Colorbar
    cbaxes = fig.add_axes([0.9, 0.2, 0.035, 0.55])             
    cbar = plt.colorbar(cref_plot, cax = cbaxes, ticks = levels_ticks)
    cbar.ax.tick_params(labelsize=15)
    plt.text(0.3, -0.1, 'K', fontsize = 21)
    
    plt.savefig("/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/png_for_gifs/3panel_wind_theta_20ms_%03d.png" % i, dpi = 100)
    plt.close(fig)
    

##Build GIF
#os.system('module load imagemagick')
os.system('convert -delay 12 -quality 100 /uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/png_for_gifs/3panel_wind_theta_20ms_*.png  /uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/gifs/3panel_wind_theta_20ms.gif')

###Delete PNGs





