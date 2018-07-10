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
import shiftedcolormap
import pandas as pd
import xarray as xr
from scipy import interpolate
from scipy import signal
import operator
import multiprocessing
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.patheffects as PathEffects
import scipy.ndimage
import matplotlib.patches as patches
import multiprocessing as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Read in with xarray
dsl_no = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group8/tom/cm1/output/tug/cm1run_150m_15ms_0000m_90sec.nc')
dsl_small = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/output/netcdf_output/tug/cm1run_150m_15ms_0500m_90sec.nc')
dsl_tall = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/output/netcdf_output/tug/cm1run_150m_15ms_2000m_90sec.nc')

dsh_no = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/output/netcdf_output/tug/cm1run_150m_25ms_0000m_90sec.nc')
dsh_small = xr.open_dataset('//uufs/chpc.utah.edu/common/home/steenburgh-group8/tom/cm1/output/tug/cm1run_150m_25ms_0500m_90sec.nc')
dsh_tall = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group8/tom/cm1/output/tug/cm1run_150m_25ms_2000m_90sec.nc')



###############################################################################
########################## Read in data #######################################
###############################################################################

ts = 0
t = 280
t_60 = np.int(t*90/60)
left = 1500
right = 2050
bottom = 0
top = 50
near = 160
far = dsh_no.ny-160
ymid = np.int(dsh_no.ny/2)
t_xy = 1



##############  Load in all data for Multiprocessing  #########################   

#U-wind   
ul_no = dsl_no.uinterp[ts:t,t_xy,near:far,left:right].values
ul_small = dsl_small.uinterp[ts:t,t_xy,near:far,left:right].values
ul_tall = dsl_tall.uinterp[ts:t,t_xy,near:far,left:right].values

uh_no = dsh_no.uinterp[ts:t,t_xy,near:far,left:right].values
uh_small = dsh_small.uinterp[ts:t,t_xy,near:far,left:right].values
uh_tall = dsh_tall.uinterp[ts:t,t_xy,near:far,left:right].values

#V-wind
vl_no = dsl_no.vinterp[ts:t,t_xy,near:far,left:right].values
vl_small = dsl_small.vinterp[ts:t,t_xy,near:far,left:right].values
vl_tall = dsl_tall.vinterp[ts:t,t_xy,near:far,left:right].values

vh_no = dsh_no.vinterp[ts:t,t_xy,near:far,left:right].values
vh_small = dsh_small.vinterp[ts:t,t_xy,near:far,left:right].values
vh_tall = dsh_tall.vinterp[ts:t,t_xy,near:far,left:right].values

##W-wind
#wl_no = dsl_no.winterp[ts:t,t_xy+1:t_xy+7,near:far,left:right].values
#wl_small = dsl_small.winterp[ts:t,t_xy+1:t_xy+7,near:far,left:right].values
#wl_tall = dsl_tall.winterp[ts:t,t_xy+1:t_xy+7,near:far,left:right].values
#
#wh_no = dsh_no.winterp[ts:t,t_xy+1:t_xy+7,near:far,left:right].values
#wh_small = dsh_small.winterp[ts:t,t_xy+1:t_xy+7,near:far,left:right].values
#wh_tall = dsh_tall.uinterp[ts:t,t_xy+1:t_xy+7,near:far,left:right].values


#x-land
land = dsl_no.xland[0,near:far,left:right]


#zs
zl_no = np.zeros((far-near,right-left))
zl_no[zl_no==0] = np.nan
zl_small = dsl_small.zs[0,near:far,left:right]
zl_tall = dsl_tall.zs[0,near:far,left:right]

zh_no = np.zeros((far-near,right-left))
zh_no[zh_no==0] = np.nan
zh_small = dsl_small.zs[0,near:far,left:right]
zh_tall = dsl_tall.zs[0,near:far,left:right]



#%%
    
    
    
    

###############################################################################
############################## Plot ###########################################
###############################################################################
    
#Colormap
colors1_t = np.array(nclcmaps.colors['BlueYellowRed'])#amwg256'])
colors_int_t = colors1_t.astype(int)
colors_t = list(colors_int_t)
cmap_wind = nclcmaps.make_cmap(colors_t, bit = True)

#cmap_wind = nclcmaps.cmap('ncl_default')

#Levels
lmin = -10
lmax = 15.01
levels = np.arange(lmin,lmax, 0.25)
levels_ticks = np.arange(lmin,lmax,5)
levels_ticks_labels = np.arange(lmin,lmax, 5).astype(int)

#Levels
wlmin = -5.25
wlmax = 5.01
wlevels = np.arange(wlmin,wlmax, 0.5)
wlevels = np.delete(wlevels, np.where(wlevels == 0))


shifted_cmap = shiftedcolormap.shiftedColorMap(cmap_wind, midpoint=1 - lmax/(lmax + abs(lmin)), name='shifted')

    
#########################  Create Fig  ########################################

    
def plotting(t):

    print(t)    

    #Create Fig
    fig = plt.figure(num=None, figsize=(18,9), facecolor='w', edgecolor='k')
    #Loop over subplots
    for run in range(1,7):
        
    
        #loop over u
        u_run_name = ['ul_no', 'ul_small', 'ul_tall','uh_no', 'uh_small', 'uh_tall']
        u_model_run = eval(u_run_name[run-1])
        
        #loop over v
        v_run_name = ['vl_no', 'vl_small', 'vl_tall','vh_no', 'vh_small', 'vh_tall']
        v_model_run = eval(v_run_name[run-1])
        
#        #loop over w
#        w_run_name = ['wl_no', 'wl_small', 'wl_tall','wh_no', 'wh_small', 'wh_tall']
#        w_model_run = eval(w_run_name[run-1])
        
        #loop over z
        z_run_name = ['zl_no', 'zl_small', 'zl_tall','zh_no', 'zh_small', 'zh_tall']
        z_model_run = eval(z_run_name[run-1])
        
            
        #############################  Plot xy  ###################################
        ax = plt.subplot(230 + run,aspect = 'equal')
        plt.subplots_adjust(left=0.06, bottom=0.15, right=0.98, top=0.95, wspace=0.06, hspace=0.01)
        ax.axis('equal')
        ax.axis('off')
        
        #Plot U-Wind
        wind_plot_xy = ax.contourf(u_model_run[t,:,:], levels, cmap = shifted_cmap, extend = 'both', alpha = 1,  zorder = 3)
        
#        #Plot W-Wind
#        w_wind_int = np.copy(np.mean(w_model_run[t:t+10,:,:,:], axis = (0,1)))
#        w_wind = np.copy(scipy.ndimage.filters.uniform_filter(w_wind_int, 20))
#        w_wind_plot = ax.contour(w_wind, wlevels, colors = 'k', extend = 'both', alpha = 1,  zorder = 12, linewidths = 1.7)
#        #clab = ax.clabel(w_wind_plot, w_wind_plot.levels, fontsize=11, inline=1, zorder = 13, fmt='%1.2f')
#        #plt.setp(clab, path_effects=[PathEffects.withStroke(linewidth=1.5, foreground="w")], zorder = 12)
#        plt.setp(w_wind_plot.collections, path_effects=[PathEffects.withStroke(linewidth=2, foreground="w")], zorder = 12)
        
        
        #Wind Vectors
        U_t_int = np.copy(u_model_run[t,:,:])
        U_t = np.copy(scipy.ndimage.filters.uniform_filter(U_t_int, 10))
        V_t_int = np.copy(v_model_run[t,:,:])
        V_t = np.copy(scipy.ndimage.filters.uniform_filter(V_t_int, 10))
        yy = np.arange(5, far-near, 30)
        xx = np.arange(0, right-left, 30)
        points = np.meshgrid(yy, xx)
        y, x = np.meshgrid(yy, xx)
        
        quiv = ax.quiver(x, y, U_t[points], V_t[points], zorder = 3,  edgecolors = 'w', linewidth = 0.5, scale = 190)


        #Land, water, terrain
        levels_water = [1.5, 2.5]
        terrain_levels = np.arange(0, 300.1, 200)
        terrain_levels_tall = np.arange(0, 2200.1, 400)
        terrain_ticks = np.arange(0,2500.1,250)
        
        #Plot Lake
        water = plt.contour(land, levels_water, alpha = 1, colors = ('blue'), zorder = 4, linewidths = 4)
        
        #Plot Terrain
        try:
            terrain = plt.contour(z_model_run, terrain_levels, alpha = 1, colors = 'w', vmin = -800, vmax = 3000, zorder = 4, linewidths = 1.75)
            
            #Plot Point at peak
            peak = np.where(z_model_run == np.max(z_model_run))
            ax.add_patch(patches.Circle((peak[1][0], peak[0][0]), 7, color = 'w', zorder = 20))

        except:
            pass
    
    
        #Draw border
        ax.add_patch(patches.Rectangle((1.5,0),right-left-3,far-near-1,fill=False, zorder = 10, linewidth = 2))
        
        #Titles
        if run == 1:
            plt.title("No Mountain", fontsize = 26, y = 0.98)
            ax.text(-0.08, 0.67, 'Low Wind', transform=ax.transAxes, fontsize = 26, rotation = 90)
        if run == 2: 
            plt.title("500m Mountain", fontsize = 26, y = 0.98)
        if run == 3: 
            plt.title("2000m Mountain", fontsize = 26, y = 0.98)
        if run == 4:
            ax.text(-0.08, 0.67, 'High Wind', transform=ax.transAxes, fontsize = 26, rotation = 90)
        if run == 6:
            props = dict(boxstyle='square', facecolor='white', alpha=1, linewidth = 2)
            ax.text(0.35, -0.2, 'Elapsed time: {:,} seconds'.format(t*90), transform=ax.transAxes, bbox = props, fontsize = 16)
            
    
        
    #Colorbars
    cbaxes = fig.add_axes([0.27, 0.09, 0.5, 0.045])           
    cbar = plt.colorbar(wind_plot_xy, orientation='horizontal', cax = cbaxes, ticks = levels_ticks)
    cbar.ax.tick_params(labelsize=19)
    plt.text(0.35, -1.85, 'U-wind  $\mathregular{[ms^{-1}]}$', fontsize = 24)
    
    #Quiver
    plt.quiverkey(quiv, X=-1.65, Y=-0.2, U=20, linewidth = 0.75, color = 'k', label='20 $\mathregular{ms^{-1}}$', labelpos='W', fontproperties={'size': '28'})
    
    #Save and Close
    plt.savefig("/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/png_for_gifs/flow_terrain_{:03d}.png".format(t), dpi = 65)
    plt.close(fig)  
    plt.switch_backend('Agg')
    
#run function to create images
pool = mp.Pool(processes = 20)
pool.map(plotting, range(t))#number of processors
pool.close()
pool.join()
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    