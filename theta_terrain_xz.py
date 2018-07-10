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
import matplotlib.ticker as ticker




    

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

t = 270
t_60 = np.int(t*90/60)
left = 1500
right = 2050
bottom = 0
top = 35
near = 160
far = dsh_no.ny-160
ymid = np.int(dsh_no.ny/2)
t_xy = 1



################################################################################
########################### Set up coordinates ################################
################################################################################

vert_resolution = 100
hor_resolution = 150
z_scale = 5

## The code below makes the data terrain following 
x1d = np.arange(0,right-left,1)
z1d = np.arange(0,top,1)

#Create 2D arrays for plotting data (first demnsion for each run)
x2d = np.zeros((6,top, right-left))
z2d = np.zeros((6,top, right-left))
run = ['dsl_no', 'dsh_no', 'dsl_small','dsh_small', 'dsl_tall', 'dsh_tall']


for j in range(6):
    model_run = eval(run[j])
    try:
        z = np.array(model_run.zs[0,ymid,left:right])/vert_resolution #Convert to gridpoints
    except:
        z = np.zeros((right-left))+0.4

    for i in range(top):
        x2d[j,i,:] = x1d
    for k in range(right-left):
        z2d[j,:,k] = z1d+z[k]

      
    
    
    

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
lmax = 20.01
levels = np.arange(lmin,lmax, 0.25)
levels_ticks = np.arange(lmin,lmax,5)
levels_ticks_labels = np.arange(lmin,lmax, 5).astype(int)

#Levels
wlmin = -5.0
wlmax = 5.01
wlevels = np.arange(wlmin,wlmax, 0.5)
wlevels = np.delete(wlevels, np.where(wlevels == 0))


shifted_cmap = shiftedcolormap.shiftedColorMap(cmap_wind, midpoint=1 - lmax/(lmax + abs(lmin)), name='shifted')

    
#########################  Create Fig  ########################################

    
for t in range(200,201):
    print(t)    

    #Create Fig
    fig = plt.figure(num=None, figsize=(16,8.5), facecolor='w', edgecolor='k')
    #Loop over subplots
    for run in range(1,7):
        
    
        #loop over runs
        run_name = ['dsl_no', 'dsh_no', 'dsl_small','dsh_small', 'dsl_tall', 'dsh_tall']
        model_run = eval(run_name[run-1])
        
            
        #############################  Plot xz ###################################
        ax = plt.subplot(320 + run,aspect = 'equal')
        plt.subplots_adjust(left=0.05, bottom=0.17, right=0.95, top=0.94, wspace=0.02, hspace=0.02)
        #plt.axis('equal')
        
        
        

        #Plot Wind
        theta = model_run.th[t,bottom:top,ymid,left:right].values-model_run.th[0,bottom:top,ymid,left:right].values #30 minute change in theta
        wind_plot_xz = plt.contourf(x2d[run-1,:,:], z2d[run-1,:,:]*z_scale, theta, levels, cmap = shifted_cmap, extend = 'both', alpha = 1,  zorder = 3)
    
    
        #Plot W-Wind
        mid = (right-left)/2
        w_wind_int = model_run.winterp[t-5:t+5,bottom:top,ymid,left:right-mid].mean(dim = ['time']).values
        w_wind = np.copy(scipy.ndimage.filters.uniform_filter(w_wind_int, 20))
        w_wind_plot = ax.contour(x2d[run-1,:,:mid], z2d[run-1,:,:mid]*z_scale, w_wind, wlevels, colors = 'k', extend = 'both', alpha = 1,  zorder = 12, linewidths = 1.3)
        clab = ax.clabel(w_wind_plot, w_wind_plot.levels, fontsize=11, inline=1, zorder = 13, fmt='%1.1f')
        plt.setp(clab, path_effects=[PathEffects.withStroke(linewidth=1.5, foreground="w")], zorder = 12)
        
        w_wind_int = model_run.winterp[t-5:t+5,bottom:top,ymid,left+mid:right].mean(dim = ['time']).values
        w_wind = np.copy(scipy.ndimage.filters.uniform_filter(w_wind_int, 20))
        w_wind_plot = ax.contour(x2d[run-1,:,mid:], z2d[run-1,:,mid:]*z_scale, w_wind, wlevels, colors = 'k', extend = 'both', alpha = 1,  zorder = 12, linewidths = 1.3)
        #plt.setp(w_wind_plot.collections, path_effects=[PathEffects.withStroke(linewidth=1.75, foreground="w")], zorder = 12)
        
        print(run)
        
        #Plot Terrain
        terrain = plt.plot(x1d, z2d[run-1,0,:]*z_scale+0.5, c = 'grey', linewidth = 4, zorder = 14)
        
        #Plot Lake
        lake = model_run.xland[0,ymid,left:right].values
        lake[lake == 1] = np.nan
        lake_plt = plt.plot(x1d, lake+1, c = 'blue', linewidth = 4, zorder = 15)
        
        #Plot Characteristics
        plt.grid(True, color = 'white', )
        ax.set_xlim([0,right-left])
        ax.set_ylim([0,top*z_scale-5*z_scale+1])

        if run == 1 or run == 3 or run == 5:
            ytick = np.arange(0,top*z_scale,5*z_scale)
            ax.set_yticks(ytick)
            yticklabs = ytick*vert_resolution/z_scale/1000.
            ax.set_yticklabels(yticklabs, fontsize = 13)
            ax.set_ylabel("Height (km)", fontsize = 16)
        else:
            ax.yaxis.set_visible(False)
        
        if run == 5 or run == 6:
            xtick =np.arange(0,right-left,10000/hor_resolution+1)
            ax.set_xticks(xtick)
            ax.set_xticklabels(xtick*hor_resolution/1000, fontsize = 13)
            ax.set_xlabel("Distance (km)", fontsize = 16)
        else:
            ax.xaxis.set_visible(False)

#        plt.axvspan(0,dsl_no.nx,color='gainsboro',lw=0)
#        if j == 3:
#            plt.xlabel('Distance (km)', fontsize = 14)
#        plt.ylabel('Height (m)', fontsize = 14)
        
        

        
        
#        #Wind Vectors
#        U_t = model_run.uinterp[t,t_xy,near:far,left:right].values
#        V_t = model_run.vinterp[t,t_xy,near:far,left:right].values
#        yy = np.arange(5, far-near, 30)
#        xx = np.arange(0, right-left, 30)
#        points = np.meshgrid(yy, xx)
#        y, x = np.meshgrid(yy, xx)
#        
#        quiv = ax.quiver(x, y, U_t[points], V_t[points], zorder = 4,  edgecolors = 'w', linewidth = 0.5, scale = 190)

        

    
        #Draw border
        #ax.add_patch(patches.Rectangle((0.3,0),right-left-2,top*z_scale-8,fill=False, zorder = 20, linewidth = 2.5))
        
        #Titles
        props = dict(boxstyle='square', facecolor='white', alpha=1, linewidth = 2)
        if run == 1:
            plt.title("Low Wind", fontsize = 23, y = 1.03)
        if run == 2: 
            plt.title("High Wind", fontsize = 23, y = 1.03)
            ax.text(1.01, 0.95, 'No Mountain'.center(15), transform=ax.transAxes,fontsize = 20, rotation = -90)
        if run == 4: 
            ax.text(1.01, 0.58, '500m\nMountain'.center(20), transform=ax.transAxes, fontsize = 20, rotation = -90)
        if run == 6:
            ax.text(0.5, -0.5, 'Elapsed time: {:,} seconds'.format(t*90), transform=ax.transAxes, bbox = props, fontsize = 16)
            ax.text(1.01, 0.58, '2000m\nMountain'.center(18), transform=ax.transAxes,fontsize = 20, rotation = -90)

            
    
        
    #Colorbars
    cbaxes = fig.add_axes([0.3, 0.075, 0.4, 0.03])           
    cbar = plt.colorbar(wind_plot_xz, orientation='horizontal', cax = cbaxes, ticks = levels_ticks)
    cbar.ax.tick_params(labelsize=15)
    plt.text(0.35, -2.2, 'U-wind  $\mathregular{(ms^{-1})}$', fontsize = 18)
    
    #Quiver
#    plt.quiverkey(quiv, X=-1.65, Y=-0.2, U=20, linewidth = 0.75, color = 'k', label='20 $\mathregular{ms^{-1}}$', labelpos='W', fontproperties={'size': '28'})
    
    #Save and Close
    plt.savefig("/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/plots/thetaterrain_xz_{:03d}.png".format(t), dpi = 150)
    plt.close(fig)  
    plt.switch_backend('Agg')
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    