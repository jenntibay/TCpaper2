"""
Plot 5-year diffference of Top15 experiments from ERA5:
a. MSLP

Created on 02 Oct 2024
by jtlopez
"""
#---------------------------------------------------------------------------------
#%%
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import matplotlib as mp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import metpy.calc as mpcalc
from matplotlib import ticker
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# %%
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### Define functions
def plot_background(ax):
	ax.add_feature(cf.LAKES.with_scale('10m'),facecolor='none', edgecolor='black',linewidth=0.5)
	ax.add_feature(cf.COASTLINE.with_scale('10m'), linewidth=0.5) 
	return ax

def plot_ticks(ax):
    ax.set_yticks(np.arange(-10, 40, 10), crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(100, 150, 20), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=""))
    ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=""))
    ax.tick_params(which='both', width=1)
    ax.tick_params(which='major', length=2)
    return ax

def plotDA(da, ax, title, cc, norm, cmap='jet', transform=ccrs.PlateCarree()):
    plot_ticks(ax)
    ax.set_title(title, fontsize=15)
    ax.text(0.83, 0.9, cc, 
            fontsize=10, fontweight='semibold',
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.grid(ls='--', alpha=0.3)
    ax.pcolormesh(da.lon, da.lat, da, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    plot_background(ax)
    ax.add_patch(plt.Rectangle((108,2), 37, 25.5, ls='--', facecolor='none', edgecolor='black', lw=1))
    return ax

def plotCBAR(cbar_ax, label, norm, cmap='jet'):
    cb1= fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), 
                  cax=cbar_ax, 
                  orientation='horizontal', 
                  extend='both',
                  aspect=30,)
    cb1.ax.tick_params(labelsize=10)
    cb1.set_label(label, fontsize=15)
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### OPEN/READ files
dir = Path('/home/jtibay/CORDEX3_TC/dynamics/')

dsexp43 = xr.open_dataset(dir / 'nc/merged_files/MERGED_exp43_6hourly.nc')

## ERA5
era5 = xr.open_dataset(dir / 'nc/merged_files/ERA5_20112015_remapbil.nc')
ref = era5.drop(['lat', 'lon'])
ref = era5.assign_coords({'lon': dsexp43.lon, 'lat': dsexp43.lat})
#MSLP
ref_mslp = ref.mslp.mean(dim='time')

exps = ['exp43', #1
        'exp31', #2
        'exp19', #3
        'exp29', #4
        'exp32', #5
        'exp17', #6
        'exp40', #7
        'exp16', #8
        'exp07', #9
        'exp28', #10
        'exp30', #11
        'exp44', #12
        'exp37', #13
        'exp08', #14
        'exp04' #15
        ]

bias_mslps = []
scorr_mslps = []

from tqdm import tqdm
for exp in tqdm(exps):
    print(f'calculating {exp}')
    ds = xr.open_dataset(dir / f'nc/merged_files/MERGED_{exp}_6hourly.nc')
    #MSLP
    model_mslp = ds.mslp.mean(dim='time')
    bias_mslp = model_mslp - ref_mslp
    scorr = xr.corr(model_mslp, ref_mslp, dim=['lon', 'lat'])
    cc_val = "{:.2f}".format(scorr.values)
    scorr_mslps.append(cc_val)
    bias_mslps.append(bias_mslp)
#----------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### set figure parameters
mp.rcParams['figure.dpi'] = 300
mp.rcParams['savefig.dpi'] = 300
mp.rcParams['font.size'] = 10
mp.rcParams['axes.linewidth'] = 1
mp.rcParams['xtick.labelsize'] = 8
mp.rcParams['ytick.labelsize'] = 8

MSLPbias_colors = ["#2d004bff", "#5c3c82ff", "#927dacff", "#c9bed5ff", "#e4deeaff",
                   "#ffffffff",
                   "#f2e2deff", "#e5c4bdff", "#cc8a7aff", "#b24f38ff", "#a82e2aff"]
cmapMSLP = mp.colors.ListedColormap(MSLPbias_colors, 11)
vmin = -5
vmax = 5
normMSLP = mcolors.Normalize(vmin=vmin, vmax=vmax)

letters = ['a', 'b', 'c', 'd', #1strow
           'e', 'f', 'g', 'h', #2ndrow
           'i', 'j', 'k', 'l', #3rdrow
           'm', 'n', 'o'] #4throw
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### Plot difference of EXPs with ERA5
mosaic = """ABCD
            EFGH
            IJKL
            MNO."""

fig, axs = plt.subplot_mosaic(mosaic, figsize=(15, 12),
                              subplot_kw={'projection': ccrs.PlateCarree()},
                              layout='constrained')

#MSLP
axList = [axs['A'], axs['B'], axs['C'], axs['D'],
          axs['E'], axs['F'], axs['G'], axs['H'],
          axs['I'], axs['J'], axs['K'], axs['L'],
          axs['M'], axs['N'], axs['O']]
for exp, scorrData, biasData, ax, lList in zip(exps, scorr_mslps, bias_mslps, 
                                               axList, letters):
    print(f'Plotting for {exp}')
    plotDA(biasData, ax, f'{lList}) {exp} - ERA5 : MSLP', 
           f'CC:{scorrData}', 
           normMSLP, cmap=cmapMSLP, 
           transform=ccrs.PlateCarree())
cbar_ax = fig.add_axes([0.775, .18, .21, .02])
plotCBAR(cbar_ax, 'MSLP Difference [hPa]', normMSLP, cmap=cmapMSLP)
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### save Fig
fig.savefig(dir / 'img/Fig05_mslp_diff.png',
            transparent=True,
            bbox_inches='tight')
#---------------------------------------------------------------------------------