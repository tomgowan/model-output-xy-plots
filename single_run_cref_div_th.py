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


#Read in with xarray
ds = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/cm1r19/run/cm1run_20ms_0000m_radiation_warmerlake.nc')


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


colors1 = np.array(nclcmaps.colors['prcp_1'])#perc2_9lev'])
colors_int = colors1.astype(int)
colors = list(colors_int)
cmap_precip = nclcmaps.make_cmap(colors, bit=True)

colors1_t = np.array(nclcmaps.colors['OceanLakeLandSnow'])
colors_int_t = colors1_t.astype(int)
colors_t = list(colors_int_t)
cmap_terrain = nclcmaps.make_cmap(colors_t, bit=True)





#%%
##############################   Plots ########################################
    
    
#for i in range(150,151):
for i in range(0,ds.dbz[:,0,0,0].size-10):
    
    secs = (i*120)+1200
 
    fig = plt.figure(num=None, figsize=(12,11.6), facecolor='w', edgecolor='k')
    
    
    ###################  Plot reflectivity  ###############################
    
    #Plot characteristics
    ax = plt.subplot(311,aspect = 'equal')
    plt.subplots_adjust(left=0.04, bottom=0.04, right=0.88, top=0.95, wspace=0, hspace=0)
    plt.axis('equal')
    plt.axis('off')
    
    #Plot variable details
    levels_cref = np.arange(10,45.01,0.1)
    levels_ticks = np.arange(10,45.02,5)
    cref_plot = plt.contourf(ds.cref[i,:,:], levels_cref, cmap = cmap_precip, alpha = 1, extend = 'max', zorder = 3) #vmin = -10,

    #Water and land details
    levels_water = [1.5, 2.5]
    levels_terrain = [0,1.5]
    xland_plt = plt.contourf(ds.xland[0,:,:], levels_water, alpha = 1, colors = ('lightsteelblue'), zorder = 2)
    xland_plt = plt.contourf(ds.xland[0,:,:], levels_terrain, alpha = 1, colors = ('gainsboro'), zorder = 1)
    
    #Labels
    sub_title = ['Composite Reflectivity']
    props = dict(boxstyle='square', facecolor='white', alpha=1)
    ax.text(20, 450, sub_title[0], fontsize = 17, bbox = props, zorder = 5)

    plt.title("20 $\mathregular{ms^{-1}}$ and No Terrain [elapsed time = %d seconds]"  % secs, fontsize = 22, y = 0.98) 

    
    #Colorbar
    cbaxes = fig.add_axes([0.9, 0.69, 0.035, 0.22])             
    cbar = plt.colorbar(cref_plot, cax = cbaxes, ticks = levels_ticks)
    cbar.ax.tick_params(labelsize=13)
    plt.text(0.1, -0.15, 'dBZ', fontsize = 18)
    
        
        
    ###################  Plot theta  ######################################
    
    
    #Plot characteristics
    ax = plt.subplot(312,aspect = 'equal')
    plt.subplots_adjust(left=0.04, bottom=0.04, right=0.88, top=0.95, wspace=0, hspace=0)
    plt.axis('equal')
    plt.axis('off')
    
        
    #Levels for theta
    levels = np.arange(264,269.001,.1)
    levels_ticks = np.arange(264,269.001,1)
    
    #Plot theta
    cref_plot = plt.contourf(ds.th[i,0,:,:], levels, cmap = cm.gist_ncar, extend = 'both', alpha = 1,  zorder = 3)

    #Plot land, water, terrain
    levels_water = [1.5, 2.5]
    levels_terrain = [0,1.5]
    terrain_levels = np.arange(-1, 3000.1, 200)
    terrain_ticks = np.arange(0,3000.1,500)
    
    water = plt.contourf(ds.xland[0,:,:], levels_water, alpha = 1, colors = ('lightsteelblue'), zorder = 2)
    land = plt.contourf(ds.xland[0,:,:], levels_terrain, alpha = 1, colors = ('gainsboro'), zorder = 1)
    try:
        terrain = plt.contourf(ds.zs[0,:,:], terrain_levels, alpha = 1, cmap = cm.Greys, vmin = -800, vmax = 3000, zorder = 1)
    except:
        pass

    
    #Labels
    sub_title = ['Surface Potential Temperature']
    props = dict(boxstyle='square', facecolor='white', alpha=1)
    ax.text(20, 450, sub_title[0], fontsize = 17, bbox = props, zorder = 5)
    
    #Colorbar
    cbaxes = fig.add_axes([0.9, 0.39, 0.035, 0.22])             
    cbar = plt.colorbar(cref_plot, cax = cbaxes, ticks = levels_ticks)
    cbar.ax.tick_params(labelsize=13)
    plt.text(0.3, -0.2, 'K', fontsize = 18)
    
        
        
        
        
        
    ###################  Plot Divergence with wind  #######################
    
    
    #Plot characteristics
    ax = plt.subplot(313,aspect = 'equal')
    plt.subplots_adjust(left=0.04, bottom=0.04, right=0.88, top=0.95, wspace=0, hspace=0)
    plt.axis('equal')
    plt.axis('off')

        
    #Levels for divergence
    levels = np.arange(-150,150.001,5)
    levels_ticks = np.arange(-150,150.001,30)
    
    #Calcualte diveregence
    u = np.array(ds.uinterp[i:i+10,0,:,:].mean(dim = 'time'))
    v = np.array(ds.vinterp[i:i+10,0,:,:].mean(dim = 'time'))
    
    du_dx = np.gradient(u, 400, axis=1) #Gradient in x
    dv_dy = np.gradient(v, 400, axis=0) #Gradient in y
    
    div = (du_dx[:,:] + dv_dy[:,:])/0.00001 #Divide by 10^-5 so units are in a normal range
    
    #Plot divergence
    div_plot = plt.contourf(div, levels, cmap = cm.bwr, extend = 'both', alpha = 1,  zorder = 3)

    #Plot land, water, terrain
    levels_water = [1.5, 2.5]
    levels_terrain = [50,51]
    terrain_levels = np.arange(-1, 3000.1, 200)
    terrain_ticks = np.arange(0,3000.1,500)
    
    water = plt.contour(ds.xland[0,:,:], levels_water, alpha = 1, colors = ('blue'), zorder = 6)
    try:
        land = plt.contour(ds.zs[0,:,:], levels_terrain, alpha = 1, colors = ('green'), zorder = 7)
    except:
        pass
    
    # Calculate Wind Barbs
    yy = np.arange(12, len(ds.uinterp[i,0,:,0])-10, 40)
    xx = np.arange(12, len(ds.uinterp[i,0,0,:])-10, 40)
    points = np.meshgrid(yy, xx)
    x, y = np.meshgrid(yy, xx)
    U = np.array(ds.uinterp[i,0,:,:])
    V = np.array(ds.vinterp[i,0,:,:])
    quiv = ax.quiver(y, x, U[points], V[points], zorder = 4, color = 'k', scale = 600)
    
    #Labels
    sub_title = ['Surface Divergence (20-min mean)']
    props = dict(boxstyle='square', facecolor='white', alpha=1)
    ax.text(20, 450, sub_title[0], fontsize = 17, bbox = props, zorder = 5)
    

    
    #Colorbar
    cbaxes = fig.add_axes([0.9, 0.08, 0.035, 0.22])             
    cbar = plt.colorbar(div_plot, cax = cbaxes, ticks = levels_ticks)
    cbar.ax.tick_params(labelsize=13)
    plt.text(-0.1, -0.2, '$\mathregular{x10^{-5}}$$\mathregular{s^{-1}}$', fontsize = 18)
    
    #Quiver
    ax.quiverkey(quiv, X=0.54, Y=-0.01, U=20, label='20 $\mathregular{ms^{-1}}$', labelpos='W', fontproperties={'size': '19'})
    
    plt.savefig("/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/png_for_gifs/single_run_cref_th_div_%03d.png" % i, dpi = 100)
    plt.close(fig)
    

##Build GIF
#os.system('module load imagemagick')
os.system('convert -delay 12 -quality 100 /uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/png_for_gifs/single_run_cref_th_div_*.png  /uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/gifs/single_run_cref_th_div.gif')

###Delete PNGs





