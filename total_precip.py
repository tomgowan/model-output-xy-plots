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
from scipy import interpolate
from scipy import signal
import operator
import multiprocessing
import smoother_leah
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.patheffects as PathEffects
import scipy.ndimage
import matplotlib.patches as patches


#Read in with xarray
ds_no = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/output/netcdf_output/tug/cm1run_150m_25ms_0000m_90sec.nc')
ds_small = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group8/tom/cm1/output/tug/cm1run_150m_25ms_0500m_90sec.nc')
ds_tall = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group8/tom/cm1/output/tug/cm1run_150m_25ms_2000m_90sec.nc')
  
#%%

###############################################################################
##################### OUTPUT CHARACTERISTICS ##################################
###############################################################################

dx = 150 #[meters]
dy = 150 #[meters]
dz = 100 #[meters] (resolution away from terrain)
ts_length = 90.0 #[seconds]
t_end = 310
t_end_60sec = 465
t_start = 160
t_start_60sec = 240
bot = 165
top = 495
left = 1000
right = 2104
hor_resolution = 150

###############################################################################
###############################################################################


################  Pull in vars and compute differences ########################

#prc_no = ds_no.rain[t_end,bot:top,left:right].values*10-ds_no.rain[t_start,bot:top,left:right].values*10
#prc_small = ds_small.rain[t_end,bot:top,left:right].values*10-ds_small.rain[t_start,bot:top,left:right].values*10
#prc_tall = ds_tall.rain[t_end,bot:top,left:right].values*10-ds_tall.rain[t_start,bot:top,left:right].values*10

prc_no = ds_no.cref[t_start:t_end,bot:top,left:right].mean('time').values
prc_small = ds_small.cref[t_start:t_end,bot:top,left:right].mean('time').values
prc_tall = ds_tall.cref[t_start:t_end,bot:top,left:right].mean('time').values

no_small = prc_no-prc_small
no_tall = prc_no-prc_tall
small_tall = prc_small-prc_tall
    
    
    

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
colors1 = np.array(nclcmaps.colors['BlWhRe'])#perc2_9lev'])
colors_int = colors1.astype(int)
colors = list(colors_int)
cmap_dif = nclcmaps.make_cmap(colors, bit=True)

colors1_t = np.array(nclcmaps.colors['prcp_1'])
colors_int_t = colors1_t.astype(int)
colors_t = list(colors_int_t)
cmap_precip = nclcmaps.make_cmap(colors_t, bit=True)





#%%
##############################   Plots ########################################
    


#Create fig
fig = plt.figure(num=None, figsize=(18,10), facecolor='w', edgecolor='k')
    
    
#Loop over subplots
for sp in range(1,7):
    subplot = 320 + sp

    #loop over runs
    run = ['ds_no', 'ds_small','ds_small','ds_tall', 'ds_tall','ds_tall']
    model_run = eval(run[sp-1])
     
    
    
#    #Levels for precip
#    lmin = 10
#    lmax = 35.01
#    levels = np.arange(lmin,lmax, 1)
#    levels_ticks = np.arange(lmin,lmax,5)
#    levels_ticks_labels = np.arange(lmin,lmax, 5).astype(int)
#    
#    #Levels for precip diff
#    lmind = -10
#    lmaxd = 10.01
#    levelsd = np.arange(lmind,lmaxd, 0.25)
#    levelsd_ticks = np.arange(lmind,lmaxd,2)
#    levelsd_ticks_labels = np.arange(lmind,lmaxd, 2).astype(int)
    
    #Levels for cref
    lmin = 10
    lmax = 55.01
    levels = np.arange(lmin,lmax, 1)
    levels_ticks = np.arange(lmin,lmax,5)
    levels_ticks_labels = np.arange(lmin,lmax, 5).astype(int)
    
    #Levels for precip diff
    lmind = -20
    lmaxd = 20.01
    levelsd = np.arange(lmind,lmaxd, 0.25)
    levelsd_ticks = np.arange(lmind,lmaxd,4)
    levelsd_ticks_labels = np.arange(lmind,lmaxd, 4).astype(int)

    #######################################################################


       
    
    #####################  Plot characteristics  ##########################
    ax = plt.subplot(subplot,aspect = 'equal')
    plt.subplots_adjust(left=0.06, bottom=0.15, right=0.98, top=0.95, wspace=0.06, hspace=0.06)
    plt.axis('equal')
    plt.axis('off')


    
    #Plot Precip
    if sp == 1:
        prc_plot = ax.contourf(prc_no, levels, cmap = cmap_precip, extend = 'both', alpha = 1,  zorder = 3)
    if sp == 3:
        prc_plot = ax.contourf(prc_small, levels, cmap = cmap_precip, extend = 'both', alpha = 1,  zorder = 3)
    if sp == 5:
        prc_plot = ax.contourf(prc_tall, levels, cmap = cmap_precip, extend = 'both', alpha = 1,  zorder = 3)
    if sp == 2:
        prcd_plot = ax.contourf(no_small, levelsd, cmap = cmap_dif, extend = 'both', alpha = 1,  zorder = 3)
    if sp == 4:
        prcd_plot = ax.contourf(no_tall, levelsd, cmap = cmap_dif, extend = 'both', alpha = 1,  zorder = 3)
    if sp == 6:
        prcd_plot = ax.contourf(small_tall, levelsd, cmap = cmap_dif, extend = 'both', alpha = 1,  zorder = 3)
        


    
    #Labels
    sub_title = ['No Mountain','0m minus 500m',
                 '500m Mountain','0m minus 2000m',
                 '2000m Mountain','500m minus 2000m']
    props = dict(boxstyle='square', facecolor='white', alpha=1)
    ax.text(20, 285, sub_title[sp-1], fontsize = 15, bbox = props, zorder = 6)
    
    #Stats
    if sp == 1:
        vol = (np.sum(prc_no)/1000000)*(model_run.ny*hor_resolution/1000*model_run.nx*hor_resolution/1000)
        text = "Max = {:.1f}mm".format(np.max(prc_no)) + "\nVolume = {:,.0f}".format(vol) + "$\mathregular{km^{3}}$"
        ax.text(15, 20, text, fontsize = 13, bbox = props, zorder = 6)
    if sp == 3:
        vol = (np.sum(prc_small)/1000000)*(model_run.ny*hor_resolution/1000*model_run.nx*hor_resolution/1000)
        text = "Max = {:.1f}mm".format(np.max(prc_small)) + "\nVolume = {:,.0f}".format(vol) + "$\mathregular{km^{3}}$"
        ax.text(15, 20, text, fontsize = 13, bbox = props, zorder = 6)
    if sp == 5:
        vol = (np.sum(prc_tall)/1000000)*(model_run.ny*hor_resolution/1000*model_run.nx*hor_resolution/1000)
        text = "Max = {:.1f}mm".format(np.max(prc_tall)) + "\nVolume = {:,.0f}".format(vol) + "$\mathregular{km^{3}}$"
        ax.text(15, 20, text, fontsize = 13, bbox = props, zorder = 6)
    if sp == 2:
        cc = np.corrcoef(np.ravel(prc_no),np.ravel(prc_small))[0,1]
        text = "CC = {:.2f}".format(cc)
        ax.text(15, 20, text, fontsize = 13, bbox = props, zorder = 6)
    if sp == 4:
        cc = np.corrcoef(np.ravel(prc_no),np.ravel(prc_tall))[0,1]
        text = "CC = {:.2f}".format(cc)
        ax.text(15, 20, text, fontsize = 13, bbox = props, zorder = 6)
    if sp == 6:
        cc = np.corrcoef(np.ravel(prc_small),np.ravel(prc_tall))[0,1]
        text = "CC = {:.2f}".format(cc)
        ax.text(15, 20, text, fontsize = 13, bbox = props, zorder = 6)
        
        
    ################  Plot land, water, terrain  ##########################
    levels_water = [1.5, 2.5]
    levels_terrain = [50,51]
    terrain_levels = np.arange(0, 2200.1, 200)
    terrain_levels_tall = np.arange(0, 2200.1, 400)
    terrain_ticks = np.arange(0,2500.1,250)
    
    #Water and land details
    levels_water = [1.5, 2.5]
    levels_terrain = [0,1.5]
    land = model_run.xland[0,bot:top,left:right].values
    water = plt.contour(model_run.xland[0,bot:top,left:right], levels_water, alpha = 1, colors = ('blue'), zorder = 4, linewidths = 3)
    

    try:
        terrain = plt.contour(model_run.zs[0,bot:top,left:right], terrain_levels, alpha = 1, cmap = cm.Greys, vmin = -800, vmax = 3000, zorder = 4)

        if sp == 2 or sp == 3:
            clab = ax.clabel(terrain, terrain.levels, fontsize=11, inline=1, zorder = 8, fmt='%1.0fm')
        else:
            clab = ax.clabel(terrain, terrain.levels[::2], fontsize=9, inline=1, zorder = 8, fmt='%1.0fm')

        plt.setp(clab, path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])
    except:
        pass
    
    #Draw border
    ax.add_patch(patches.Rectangle((1.5,0),model_run.nx*0.5-3,model_run.ny*0.5,fill=False, zorder = 10, linewidth = 2))


#Colorbars
cbaxes = fig.add_axes([0.13, 0.09, 0.3, 0.035])           
cbar = plt.colorbar(prc_plot, orientation='horizontal', cax = cbaxes, ticks = levels_ticks)
cbar.ax.tick_params(labelsize=15)
plt.text(0.45, -1.8, 'mm', fontsize = 22)

cbaxes = fig.add_axes([0.61, 0.09, 0.3, 0.035])           
cbar = plt.colorbar(prcd_plot, orientation='horizontal', cax = cbaxes, ticks = levelsd_ticks)
cbar.ax.tick_params(labelsize=15)
plt.text(0.45, -1.8, 'mm', fontsize = 22)

plt.savefig("/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/plots/6panel_cref_high_wind.png", dpi = 200)
plt.close(fig)
    
    
    
    