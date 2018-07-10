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


#%%

### Read in with netcdf
#cm1_file = "/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/cm1out_lake_200m_skinny_lake.nc"
#fh = Dataset(cm1_file)
#
#cref = fh.variables['cref'][:,:,:]
#xland  = fh.variables['xland'][:]
##zs  = fh.variables['zs'][:]


#Read in with xarray
ds= xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/output/cm1out_200m_downstream_mtn_idealized.nc')
ds2= xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/output/cm1out_200m_downstream_small_mtn_idealized.nc')

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
colors1 = np.array(nclcmaps.colors['prcp_1'])#perc2_9lev'])
colors_int = colors1.astype(int)
colors = list(colors_int)
cmap_precip = nclcmaps.make_cmap(colors, bit=True)

colors1_t = np.array(nclcmaps.colors['MPL_Greys'])
colors_int_t = colors1_t.astype(int)
colors_t = list(colors_int_t)
cmap_terrain = nclcmaps.make_cmap(colors_t, bit=True)


#%%

##Chose Variable
var = np.array(ds.qi.mean(+ds.qc)

#%%
##############################   Plots ########################################
    
    
for i in range(0,len(ds.cref[:,0,0])):
    
    secs = i*120
    
    #Set up plot
    fig = plt.figure(num=None, figsize=(12,8), facecolor='w', edgecolor='k')
    ax = plt.subplot(121,aspect = 'equal')
    plt.subplots_adjust(left=0.04, bottom=0.01, right=0.9, top=0.9, wspace=0, hspace=0)
    plt.axis('equal')
    plt.axis('off')
    
    #Create grid
    x = np.arange(0,1300,1)
    y = np.arange(0,400,1)
    
    #Levels for CREF
    levels_cref = np.arange(5,40.01,0.5)
    levels_cref_ticks = np.arange(5,40.01,5)
    
    #Plot reflectivity
    cref_plot = plt.contourf(ds.cref[i,:,:], levels_cref, cmap = cmap_precip, extend = 'max', alpha = 1,  zorder = 3)

    #Plot land, water, terrain
    levels_water = [1.5, 2.5]
    levels_terrain = [0,1.5]
    terrain_levels = np.arange(-1, 3000.1, 200)
    terrain_ticks = np.arange(0,3000.1,500)
    
    water = plt.contourf(ds.xland[0,:,:], levels_water, alpha = 1, colors = ('lightsteelblue'), zorder = 2)
    land = plt.contourf(ds.xland[0,:,:], levels_terrain, alpha = 1, colors = ('gainsboro'), zorder = 1)
    terrain = plt.contourf(ds.zs[0,:,:], terrain_levels, alpha = 1, cmap = cm.Greys, vmin = -800, vmax = 3000, zorder = 1)
    
    #Title
    plt.title("Composite Reflectivity [dBZ] (elapsed time = %d seconds)"  % secs, fontsize = 21, y = 0.95) 

    #Colorbar    
    cbaxes = fig.add_axes([0.93, 0.08, 0.03, 0.8])             
    cbar = plt.colorbar(cref_plot, cax = cbaxes, ticks = levels_cref_ticks)
    cbar.ax.tick_params(labelsize=15)
    
    #Labels
    sub_title = '2500m Mountain'
    props = dict(boxstyle='square', facecolor='white', alpha=1)
    ax.text(20, 356, sub_title, fontsize = 18, bbox = props, zorder = 5)

    plt.savefig("/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/png_for_gifs/downstream_tall_mtn_200m_%03d.png" % i)
    plt.close(fig)

    
    
    

##Build GIF
os.system('module load imagemagick')
os.system('/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/png_for_gifs/downstream_tall_mtn_200m_*.png ../gifs/downstream_tall_mtn_cref.gif')

###Delete PNGs





