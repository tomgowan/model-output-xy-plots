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
ds_low = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group8/tom/cm1/output/tug/cm1run_150m_15ms_0000m_90sec.nc')
ds_high = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/output/netcdf_output/tug/cm1run_150m_25ms_0000m_90sec.nc')

#%%

###############################################################################
##################### OUTPUT CHARACTERISTICS ##################################
###############################################################################
start_ts = 260 #Starting time step
end_ts = 261 #Starting time step
ts = start_ts
height = 1
dx = 150 #[meters]
dy = 150 #[meters]
dz = 100 #[meters] (resolution away from terrain)
ts_length = 90.0 #[seconds]
ts_num = 25
h_num = 3
smooth = 25
smooth_f = 25
###############################################################################
###############################################################################


################  Function to calculate frontogenesis #########################

#Kinematic Frontogenesis
def kin_fronto(model_run, ts, height, dx, dy, ts_len, h_num):
    # Fetch the fields we need
    u = model_run.uinterp[ts:ts+ts_num,height:height+h_num,:,:].mean(dim = ('time','nk')).values
    u[np.isnan(u)] = 0
    u = np.copy(scipy.ndimage.filters.uniform_filter(u, smooth_f))
    
    v = model_run.vinterp[ts:ts+ts_num,height:height+h_num,:,:].mean(dim = ('time','nk')).values
    v[np.isnan(v)] = 0
    v = np.copy(scipy.ndimage.filters.uniform_filter(v, smooth_f))

    theta = model_run.th[ts:ts+ts_num,height:height+h_num,:,:].mean(dim = ('time','nk')).values
    theta[np.isnan(theta)] = 0
    theta = np.copy(scipy.ndimage.filters.uniform_filter(theta, smooth_f))
    
    #Calculate gradients
    theta_gradient = np.sqrt((np.gradient(theta, dx, axis=1))**2 + (np.gradient(theta, dy, axis=0))**2)
    zonal_gradient = (-1 * np.gradient(theta, dx, axis=1)) * ((np.gradient(u, dx, axis=1) * np.gradient(theta, dx, axis=1)) + (np.gradient(v, dx, axis=1) * np.gradient(theta, dy, axis=0)))
    meridional_gradient = (-1 * np.gradient(theta, dy, axis=0)) * ((np.gradient(u, dy, axis=0) * np.gradient(theta, dx, axis=1)) + (np.gradient(v, dy, axis=0) * np.gradient(theta, dy, axis=0)))

    #Compute frontogenesis
    F2K = (1 / theta_gradient) * (zonal_gradient + meridional_gradient)
    
    #Change units to [K/(10km*h)]
    F2K = F2K * 3600 * 10000
    
    return F2K


#Diabatic Frontogenesis
def dia_fronto(model_run, ts, height, dx, dy, ts_len, h_num):
    # Fetch the fields we need
    theta = model_run.th[ts:ts+ts_num,height:height+h_num,:,:].mean(dim = ('time','nk')).values
    theta[np.isnan(theta)] = 0
    theta = np.copy(scipy.ndimage.filters.uniform_filter(theta, smooth_f))
    
    ptb_hidiff = model_run.ptb_hidiff[ts:ts+ts_num,height:height+h_num,:,:].mean(dim = ('time','nk')).values
    ptb_hidiff [np.isnan(ptb_hidiff )] = 0
    ptb_hidiff  = np.copy(scipy.ndimage.filters.uniform_filter(ptb_hidiff , smooth_f))
    
    ptb_vidiff = model_run.ptb_vidiff[ts:ts+ts_num,height:height+h_num,:,:].mean(dim = ('time','nk')).values
    ptb_vidiff[np.isnan(ptb_vidiff)] = 0
    ptb_vidiff = np.copy(scipy.ndimage.filters.uniform_filter(ptb_vidiff, smooth_f))
    
    ptb_hturb = model_run.ptb_hturb[ts:ts+ts_num,height:height+h_num,:,:].mean(dim = ('time','nk')).values
    ptb_hturb[np.isnan(ptb_hturb)] = 0
    ptb_hturb = np.copy(scipy.ndimage.filters.uniform_filter(ptb_hturb, smooth_f))
    
    ptb_vturb = model_run.ptb_vturb[ts:ts+ts_num,height:height+h_num,:,:].mean(dim = ('time','nk')).values
    ptb_vturb[np.isnan(ptb_vturb)] = 0
    ptb_vturb = np.copy(scipy.ndimage.filters.uniform_filter(ptb_vturb, smooth_f))
    
    ptb_mp = model_run.ptb_mp[ts:ts+ts_num,height:height+h_num,:,:].mean(dim = ('time','nk')).values
    ptb_mp[np.isnan(ptb_mp)] = 0
    ptb_mp = np.copy(scipy.ndimage.filters.uniform_filter(ptb_mp, smooth_f))
    
    ptb_rdamp = model_run.ptb_rdamp[ts:ts+ts_num,height:height+h_num,:,:].mean(dim = ('time','nk')).values
    ptb_rdamp[np.isnan(ptb_rdamp)] = 0
    ptb_rdamp = np.copy(scipy.ndimage.filters.uniform_filter(ptb_rdamp, smooth_f))
    
    ptb_rad = model_run.ptb_rad[ts:ts+ts_num,height:height+h_num,:,:].mean(dim = ('time','nk')).values
    ptb_rad[np.isnan(ptb_rad)] = 0
    ptb_rad = np.copy(scipy.ndimage.filters.uniform_filter(ptb_rad, smooth_f))
    
    ptb_div = model_run.ptb_div[ts:ts+ts_num,height:height+h_num,:,:].mean(dim = ('time','nk')).values
    ptb_div[np.isnan(ptb_div)] = 0
    ptb_div = np.copy(scipy.ndimage.filters.uniform_filter(ptb_div, smooth_f))
    
    ptb_diss = model_run.ptb_diss[ts:ts+ts_num,height:height+h_num,:,:].mean(dim = ('time','nk')).values
    ptb_diss[np.isnan(ptb_diss)] = 0
    ptb_diss = np.copy(scipy.ndimage.filters.uniform_filter(ptb_diss, smooth_f))
    
    
    diabatic = np.copy(ptb_hidiff + ptb_vidiff + ptb_hturb + ptb_vturb + ptb_mp + ptb_rdamp + ptb_rad + ptb_div + ptb_diss)


    #Calculate gradients
    theta_gradient = np.sqrt((np.gradient(theta, dx, axis=1))**2 + (np.gradient(theta, dy, axis=0))**2)
    zonal_gradient = (-1 * np.gradient(theta, dx, axis=1) * np.gradient(diabatic, dx, axis=1))
    meridional_gradient = (-1 * np.gradient(theta, dy, axis=0) * np.gradient(diabatic, dy, axis=0))

    #Compute frontogenesis
    F2D = (-1 / theta_gradient) * (zonal_gradient + meridional_gradient)
    
    #Change units to [K/(10km*h)]
    F2D = F2D * 3600 * 10000
    
    return F2D

##################### Calculate Frontogenesis #########################

#Call frontogenesis function
run = ['ds_low', 'ds_high']
#Low wind
kin_fronto_lows = kin_fronto(eval(run[0]), ts, height, dx, dy, ts, h_num)
#kin_fronto_low[np.isnan(kin_fronto_low)] = 0
#kin_fronto_lows = np.copy(scipy.ndimage.filters.uniform_filter(kin_fronto_low, smooth_f))

dia_fronto_lows = dia_fronto(eval(run[0]), ts, height, dx, dy, ts, h_num)
#dia_fronto_low[np.isnan(dia_fronto_low)] = 0
#dia_fronto_lows = np.copy(scipy.ndimage.filters.uniform_filter(dia_fronto_low, smooth_f))

total_fronto_low = np.array(kin_fronto_lows + dia_fronto_lows)


#High wind
kin_fronto_highs = kin_fronto(eval(run[1]), ts, height, dx, dy, ts_num, h_num)
#kin_fronto_high[np.isnan(kin_fronto_high)] = 0
#kin_fronto_highs = np.copy(scipy.ndimage.filters.uniform_filter(kin_fronto_high, smooth_f))

dia_fronto_highs = dia_fronto(eval(run[1]), ts, height, dx, dy, ts_num, h_num)
#dia_fronto_high[np.isnan(dia_fronto_high)] = 0
#dia_fronto_highs = np.copy(scipy.ndimage.filters.uniform_filter(dia_fronto_high, smooth_f))

total_fronto_high = np.array(kin_fronto_highs + dia_fronto_highs)


    
    
    

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
cmap_var = nclcmaps.make_cmap(colors, bit=True)

colors1_t = np.array(nclcmaps.colors['MPL_Greys'])
colors_int_t = colors1_t.astype(int)
colors_t = list(colors_int_t)
cmap_terrain = nclcmaps.make_cmap(colors_t, bit=True)





#%%
##############################   Plots ########################################
    
#Loop over all times for frontogenesis plots  
for ts in range(start_ts, end_ts):
    
    secs = (ts*ts_length) #Current model time

    #Create fig
    fig = plt.figure(num=None, figsize=(18,10), facecolor='w', edgecolor='k')
    
    #Loop over subplots
    for sp in range(1,7):
        subplot = 320 + sp

        #loop over runs
        run = ['ds_low', 'ds_high','ds_low','ds_high','ds_low','ds_high']
        model_run = eval(run[sp-1])
        
    
        
        
        
        #Levels for frontogenesis
        lmin = -15
        lmax = 15.01
        levels = np.arange(lmin,lmax, 0.5)
        levels_ticks = np.arange(lmin,lmax,5)
        levels_ticks_labels = np.arange(lmin,lmax, 5).astype(int)
        
        #Levels for theta
        tlmin = 260
        tlmax = 270
        tlevels = np.arange(tlmin,tlmax, 1)
        tlevels_ticks = np.arange(tlmin,tlmax,1)
        tlevels_ticks_labels = np.arange(tlmin,tlmax, 1).astype(int)

        
        #######################################################################
        
        
        
        ##################### Calculate Wind Barbs ############################
        
        yy = np.arange(12, model_run.ny-10, 60)
        xx = np.arange(12, model_run.nx-10, 60)
        points = np.meshgrid(yy, xx)
        x, y = np.meshgrid(yy, xx)
        
        if sp == 1 or sp == 3 or sp == 5:
            U = np.array(model_run.uinterp[ts:ts+ts_num,height:height+h_num,:,:].mean(dim = ('time','nk')))
            V = np.array(model_run.vinterp[ts:ts+ts_num,height:height+h_num,:,:].mean(dim = ('time','nk')))
            
        if sp == 2 or sp == 4 or sp == 6:
            U = np.array(model_run.uinterp[ts:ts+ts_num,height:height+h_num,:,:].mean(dim = ('time','nk')))
            V = np.array(model_run.vinterp[ts:ts+ts_num,height:height+h_num,:,:].mean(dim = ('time','nk')))

        
        #######################################################################

        

           
        
        #####################  Plot characteristics  ##########################
        ax = plt.subplot(subplot,aspect = 'equal')
        plt.subplots_adjust(left=0.06, bottom=0.15, right=0.98, top=0.95, wspace=0.06, hspace=0.06)
        plt.axis('equal')
        plt.axis('off')

        
        #Plot frontogenesis
        if sp == 1:
            fronto_plot = ax.contourf(kin_fronto_lows, levels, cmap = cmap_var, extend = 'both', alpha = 1,  zorder = 1)
        if sp == 2:
            fronto_plot = ax.contourf(kin_fronto_highs, levels, cmap = cmap_var, extend = 'both', alpha = 1,  zorder = 1)
        if sp == 3:
            fronto_plot = ax.contourf(dia_fronto_lows, levels, cmap = cmap_var, extend = 'both', alpha = 1,  zorder = 1)
        if sp == 4:
            fronto_plot = ax.contourf(dia_fronto_highs, levels, cmap = cmap_var, extend = 'both', alpha = 1,  zorder = 1)
        if sp == 5:
            fronto_plot = ax.contourf(total_fronto_low, levels, cmap = cmap_var, extend = 'both', alpha = 1,  zorder = 1)
        if sp == 6:
            fronto_plot = ax.contourf(total_fronto_high, levels, cmap = cmap_var, extend = 'both', alpha = 1,  zorder = 1)
            
        #Plot Winds
        quiv = ax.quiver(y, x, U[points], V[points], zorder = 3, color = 'k', width = 0.0017, scale = 450)
        
        #Plot Theta
        if sp == 1 or sp == 3 or sp == 5:
            theta = model_run.th[ts:ts+ts_num,height+h_num+2,:,:].mean(dim = ('time')).values
            theta = np.copy(scipy.ndimage.filters.uniform_filter(theta, smooth))
            theta_plot = plt.contour(theta, tlevels, alpha = 1, colors = ('grey'), zorder = 8, linewidths = 1.5)
            clab = plt.clabel(theta_plot, theta_plot.levels, fontsize=11, inline=1, zorder = 8, fmt='%1.0f')
            plt.setp(clab, path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])
            plt.setp(theta_plot.collections, path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])

        if sp == 2 or sp == 4 or sp == 6:
            theta = model_run.th[ts:ts+ts_num,height+h_num+2,:,:].mean(dim = ('time')).values
            theta = np.copy(scipy.ndimage.filters.uniform_filter(theta, smooth))
            theta_plot = ax.contour(theta, tlevels, alpha = 1, colors = ('grey'), zorder = 8, linewidths = 1.5)
            clab = ax.clabel(theta_plot, theta_plot.levels, fontsize=11, inline=1, zorder = 8, fmt='%1.0f')
            plt.setp(clab, path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])
            plt.setp(theta_plot.collections, path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])

        
        #Labels
        sub_title = ['Low Wind','High Wind',
                     'Low Wind','High Wind',
                     'Low Wind','High Wind']
        props = dict(boxstyle='square', facecolor='white', alpha=1)
        ax.text(35, 590, sub_title[sp-1], fontsize = 15, bbox = props, zorder = 16)
        
            
            
        ################  Plot land, water, terrain  ##########################
        levels_water = [1.5, 2.5]
        levels_terrain = [50,51]
        terrain_levels = np.arange(-1, 3000.1, 200)
        terrain_ticks = np.arange(0,3000.1,500)
        
        water = plt.contour(model_run.xland[0,:,:], levels_water, alpha = 1, colors = ('k'), zorder = 2, linewidths = 2)
        try:
            land = plt.contour(model_run.zs[0,:,:], levels_terrain, alpha = 1, colors = ('green'), zorder = 7)
            terrain = plt.contour(model_run.zs[0,:,:], terrain_levels, alpha = 1, cmap = cm.Greys, vmin = -800, vmax = 3000, zorder = 1)
        except:
            pass
        
        #Draw border
        ax.add_patch(patches.Rectangle((5,0),model_run.nx-9,model_run.ny+5,fill=False, zorder = 10, linewidth = 2))

    #Labels
    plt.text(-2500, 2000, 'Kinematic', fontsize = 24, rotation=90)
    plt.text(-2500, 1200, 'Diabatic', fontsize = 24, rotation=90)
    plt.text(-2500, 400, 'Total', fontsize = 24, rotation=90)
    
    #Colorbar
    cbaxes = fig.add_axes([0.32, 0.09, 0.4, 0.035])           
    cbar = plt.colorbar(fronto_plot, orientation='horizontal', cax = cbaxes, ticks = levels_ticks)
    cbar.ax.tick_params(labelsize=17)
    plt.text(0.15, -2, 'Frontogenesis  [$\mathregular{K(10km*h)^{-1}}$]', fontsize = 22)
    

    
    #Quiver
    ax.quiverkey(quiv, X=0.68, Y=-0.19, U=20, label='20 $\mathregular{ms^{-1}}$', labelpos='W', fontproperties={'size': '25'})
    
    plt.savefig("/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/plots/2panel_frontogenesis_dia_%03d.png" % ts, dpi = 200)
    plt.close(fig)
    
    
    
    