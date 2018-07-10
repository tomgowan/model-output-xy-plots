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
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.patheffects as PathEffects
import scipy.ndimage
import matplotlib.patches as patches
import multiprocessing as mp

#Read in with xarray
dsl = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group8/tom/cm1/output/tug/cm1run_150m_15ms_0000m_90sec.nc')
dsh = xr.open_dataset('/uufs/chpc.utah.edu/common/home/steenburgh-group7/tom/cm1/output/netcdf_output/tug/cm1run_150m_25ms_0000m_90sec.nc')



###############################################################################
###############################################################################   
##########################   Set up Trajectories ##############################
###############################################################################
###############################################################################


#Dimension size variables
num_x = dsl.uinterp[0,0,0,:].size
num_y = dsl.uinterp[0,0,:,0].size
num_z = dsl.uinterp[0,:,0,0].size

x = np.arange(0,num_x,1)
y = np.arange(0,num_y,1)
z = np.arange(0,num_z,1)



###############################################################################
##################### INFO TO CALCULATE SEEDS #################################
#############  These are variables the user changes  ##########################
###############################################################################
#Backward trajectories
parcel_spread = 13
num_seeds_z = 151 #Up to 5000m (3 seeds every vertical grid point)
num_seeds_y = np.int(dsl.ny/parcel_spread) 
num_seeds_x = np.int(dsl.nx/parcel_spread) 
time_steps = 300#ds.uinterp[:,0,0,0].size-2 #Number of time steps to run trajectories forward
start_time_step = 10 #Starting time step
hor_resolution = 150 #[meters]
vert_resolution = 100 #[meters] (resolution away from terrain)
time_step_length = 90.0 #[seconds]
###############################################################################
###############################################################################


















###############################################################################
#############################   PLOTS   #######################################
###############################################################################



#########  Create Grid ########
## The code below makes the data terrain following 
ymid = np.int(dsl.ny/2)
x1d = np.arange(0,dsl.nx,1)
y1d = np.arange(0,dsl.nz,1)
try:
    z = np.array(dsl.zs[0,ymid,:dsl.nx])/1000*10 #Div by 1000 to go to m and mult by 30 to match y dim
except:
    z = np.zeros((dsl.nx))


#Create 2D arrays for plotting data
x2d = np.zeros((dsl.nz, dsl.nx))
y2d = np.zeros((dsl.nz, dsl.nx))

for i in range(dsl.nz):
    x2d[i,:] = x1d
for j in range(dsl.nx):
    y2d[:,j] = y1d+z[j]
        
#Variables for plotting
xmin = 0
xmax = dsl.nx
xlen = xmax-xmin

zmin = 0
zmax = dsl.nz

#Variables from output to plot
land = dsl.xland[0,:,:].values
try:
    zs = dsl.zs[0,:,:].values
except:
    zs = np.zeros((dsl.ny,dsl.nx))
    
#################### Get meterolgical variables ###############################
ts = 200
te = time_steps

#Reflectivity
crefl = dsl.cref[ts:te,:,:].values

#Theta
thetal = dsl.th[ts:te,4:8,:,:].mean(dim = ('nk')).values

#Wind Barbs 

yy = np.arange(30, dsl.ny, 60)
xx = np.arange(30, dsl.nx, 60)
points = np.meshgrid(yy, xx)
x, y = np.meshgrid(yy, xx)

Ul = np.array(dsl.uinterp[ts:te,0:4,:,:].mean(dim = ('nk')))
Vl = np.array(dsl.vinterp[ts:te,0:4,:,:].mean(dim = ('nk')))



#Reflectivity
crefh = dsh.cref[ts:te,:,:].values

#Theta
thetah = dsh.th[ts:te,4:8,:,:].mean(dim = ('nk')).values

#Wind Barbs 

yy = np.arange(30, dsh.ny, 60)
xx = np.arange(30, dsh.nx, 60)
points = np.meshgrid(yy, xx)
x, y = np.meshgrid(yy, xx)

Uh = np.array(dsh.uinterp[ts:te,0:4,:,:].mean(dim = ('nk')))
Vh = np.array(dsh.vinterp[ts:te,0:4,:,:].mean(dim = ('nk')))



#%%
#Read in colormap and put in proper format
colors1 = np.array(nclcmaps.colors['amwg256'])#perc2_9lev'])
colors_int = colors1.astype(int)
colors = list(colors_int)
cmap_dth = nclcmaps.make_cmap(colors, bit=True)


#Read in colormap and put in proper format
colors1 = np.array(nclcmaps.colors['WhiteBlueGreenYellowRed'])#perc2_9lev'])
colors_int = colors1.astype(int)
colors = list(colors_int)
cmap_th = nclcmaps.make_cmap(colors, bit=True)

colors1 = np.array(nclcmaps.colors['prcp_1'])#perc2_9lev'])
colors_int = colors1.astype(int)
colors = list(colors_int)
cmap_precip = nclcmaps.make_cmap(colors, bit=True)


###############################################################################
###############################################################################



###############################################################################
#############################   Plot  #########################################
###############################################################################
#Use multiple processors to create images
def plotting(i): 


#for i in range(time_steps-6):   
#for i in range(0,1): 
    secs = (i*time_step_length)
    fig = plt.figure(num=None, figsize=(13,11), facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.07, bottom=0.22, right=0.94, top=0.9, wspace=0.12, hspace=0.2)
    
    #Levels for cref
    creflmin = 10
    creflmax = 55.01
    creflevels = np.arange(creflmin,creflmax, 0.05)
    creflevels_ticks = np.arange(creflmin,creflmax,5)
    creflevels_ticks_labels = np.arange(creflmin,creflmax, 5).astype(int)
    
    #Levels for theta
    tlmin = 265
    tlmax = 269
    tlevels = np.arange(tlmin,tlmax, 0.75)
    tlevels_ticks = np.arange(tlmin,tlmax,1)
    tlevels_ticks_labels = np.arange(tlmin,tlmax, 1).astype(int)
    
    
    ax = plt.subplot(211,aspect = 'equal')
    plt.title("Reflectivity, Potential Temperature, and Wind", fontsize = 26, y = 1.05) 

    
    #Plot reflectivity
    cref_plot = plt.contourf(crefl[i,:,:], creflevels, cmap = cmap_precip, extend = 'max', alpha = 1,  zorder = 3)
    
    #Plot Theta
    if i > 3: #Take mean of 4 time steps
        theta_meanl = np.mean(thetal[i-2:i+2,:,:], axis = 0)
    else:
        theta_meanl = thetal[i,:,:]
    theta_tl = np.copy(scipy.ndimage.filters.uniform_filter(theta_meanl[:,:], 30))
    theta_plotl = plt.contour(theta_tl, tlevels, alpha = 1, cmap = cm.coolwarm, zorder = 5, linewidths = 1.7)
    #clab = plt.clabel(theta_plot, theta_plot.levels, fontsize=11, inline=1, zorder = 5, fmt='%1.0f')
    #plt.setp(clab, path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])
    plt.setp(theta_plotl.collections, path_effects=[PathEffects.withStroke(linewidth=2.5, foreground="w")])
    
    #Plot Winds
    if i > 3: #Take mean of 4 time steps
        U_tl = np.copy(np.mean(Ul[i-2:i+2,:,:], axis = 0))
        V_tl = np.copy(np.mean(Vl[i-2:i+2,:,:], axis = 0))
    else:
        U_tl = np.copy(Ul[i,:,:])
        V_tl = np.copy(Vl[i,:,:])
        
    #quiv = ax.barbs(y, x, U_tl[points], V_tl[points], zorder = 4, color = 'k', linewidth = 0.0017, scale = 800)
    quiv = ax.quiver(y, x, U_tl[points], V_tl[points], zorder = 4,  edgecolors = 'w', width = 0.0019, scale = 550)
    
        
    #Water and land details
    levels_water = [1.5, 2.5]
    levels_terrain = [0,1.5]
    xland_plt = plt.contourf(land, levels_water, alpha = 1, colors = ('lightsteelblue'), zorder = 2)
    xland_plt = plt.contourf(land, levels_terrain, alpha = 1, colors = ('gainsboro'), zorder = 1)
    
    #Terrain
    terrain_levels = np.arange(-1, 3000.1, 200)
    terrain = plt.contourf(zs, terrain_levels, alpha = 1, cmap = cm.Greys, vmin = -800, vmax = 3000, zorder = 1)
    
    #Plot Characteristics
    plt.xticks(np.arange(0,num_x,30000/hor_resolution))
    plt.yticks(np.arange(0,num_y,15000/hor_resolution))
    ax.set_xticklabels(np.arange(0,num_x*hor_resolution/1000,30), fontsize = 16)
    ax.set_yticklabels(np.arange(0,num_x*hor_resolution/1000,15), fontsize = 16)
    plt.xlim([0,num_x])
    plt.ylim([0,num_y])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.ylabel('Distance (km)', fontsize = 22)

    
    #Labels
    sub_title = ['Low Wind']
    props = dict(boxstyle='square', facecolor='white', alpha=1)
    ax.text(0.012, 0.9, sub_title[0], bbox = props, fontsize = 20, zorder = 5, transform=ax.transAxes)
    
    
    
    
    
    
    
    ax = plt.subplot(212,aspect = 'equal')


    
    #Plot reflectivity
    cref_plot = plt.contourf(crefh[i,:,:], creflevels, cmap = cmap_precip, extend = 'max', alpha = 1,  zorder = 3)
    
    #Plot Theta
    if i > 3: #Take mean of 4 time steps
        theta_meanh = np.mean(thetah[i-2:i+2,:,:], axis = 0)
    else:
        theta_meanh = thetah[i,:,:]
    theta_th = np.copy(scipy.ndimage.filters.uniform_filter(theta_meanh[:,:], 30))
    theta_ploth = plt.contour(theta_th, tlevels, alpha = 1, cmap = cm.coolwarm, zorder = 5, linewidths = 1.7)
    #clab = plt.clabel(theta_plot, theta_plot.levels, fontsize=11, inline=1, zorder = 5, fmt='%1.0f')
    #plt.setp(clab, path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])
    plt.setp(theta_ploth.collections, path_effects=[PathEffects.withStroke(linewidth=2.5, foreground="w")])
    
    #Plot Winds
    if i > 3: #Take mean of 4 time steps
        U_th = np.copy(np.mean(Uh[i-2:i+2,:,:], axis = 0))
        V_th = np.copy(np.mean(Vh[i-2:i+2,:,:], axis = 0))
    else:
        U_th = np.copy(Uh[i,:,:])
        V_th = np.copy(Vh[i,:,:])
        
    quiv = ax.quiver(y, x, U_th[points], V_th[points], zorder = 4,  edgecolors = 'w', width = 0.0019, scale = 550)
    
        
    #Water and land details
    levels_water = [1.5, 2.5]
    levels_terrain = [0,1.5]
    xland_plt = plt.contourf(land, levels_water, alpha = 1, colors = ('lightsteelblue'), zorder = 2)
    xland_plt = plt.contourf(land, levels_terrain, alpha = 1, colors = ('gainsboro'), zorder = 1)
    
    #Terrain
    terrain_levels = np.arange(-1, 3000.1, 200)
    terrain = plt.contourf(zs, terrain_levels, alpha = 1, cmap = cm.Greys, vmin = -800, vmax = 3000, zorder = 1)
    
    #Plot Characteristics
    plt.xticks(np.arange(0,num_x,30000/hor_resolution))
    plt.yticks(np.arange(0,num_y,15000/hor_resolution))
    ax.set_xticklabels(np.arange(0,num_x*hor_resolution/1000,30), fontsize = 15)
    ax.set_yticklabels(np.arange(0,num_x*hor_resolution/1000,15), fontsize = 15)
    plt.xlim([0,num_x])
    plt.ylim([0,num_y])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.ylabel('Distance (km)', fontsize = 22)
    plt.xlabel('Distance (km)', fontsize = 22)

    
    #Labels
    sub_title = ['High Wind']
    props = dict(boxstyle='square', facecolor='white', alpha=1)
    ax.text(0.012, 0.9, sub_title[0], bbox = props, fontsize = 20, zorder = 5, transform=ax.transAxes)
    
    
    
    #Colorbar
    cbaxes = fig.add_axes([0.25, 0.08, 0.5, 0.035])
    cbar = plt.colorbar(cref_plot, orientation='horizontal', cax = cbaxes, ticks = creflevels_ticks)
    cbar.ax.set_yticklabels(creflevels_ticks_labels)
    cbar.ax.tick_params(labelsize=16)
    plt.text(0.49, -1.7, 'dBZ', fontsize = 23)
    

    
    

    
     #Quiver
    ax.quiverkey(quiv, X=0.21, Y=-0.25, U=20, label='20 $\mathregular{ms^{-1}}$', labelpos='W', fontproperties={'size': '22'})
    
    ax.text(0.66, -0.25, 'Elapsed time: {:,} seconds'.format((i+ts)*90), transform=ax.transAxes, bbox = props, fontsize = 19)

    plt.savefig("/uufs/chpc.utah.edu/common/home/u1013082/public_html/phd_plots/cm1/png_for_gifs/cref_th_wind_0000m_25ms_90s_%03d.png" % np.int(i+ts), dpi = 80)
    plt.close(fig)
    print(i)
#
plt.switch_backend('Agg')
#  
#  
#run function to create images
pool = mp.Pool(processes = 30)
pool.map(plotting, range(te-ts))#number of processors
pool.close()
pool.join()
        



