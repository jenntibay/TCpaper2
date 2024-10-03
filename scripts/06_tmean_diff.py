"""
Plot 5-year diffference of Top15 experiments from ERA5:
e. Mean Temperature, averaged over four levels 
   (8850-, 700-, 500- and 300-hPa)

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

def calculate_tmean(da):
    da_ta850 = da['ta'].sel(plev=850).mean(dim='time')
    da_ta850 = np.subtract(da_ta850, 273.15)
    da_ta700 = da['ta'].sel(plev=700).mean(dim='time')
    da_ta700 = np.subtract(da_ta700, 273.15)
    da_ta500 = da['ta'].sel(plev=500).mean(dim='time')
    da_ta500 = np.subtract(da_ta500, 273.15)
    da_ta300 = da['ta'].sel(plev=300).mean(dim='time')
    da_ta300 = np.subtract(da_ta300, 273.15)
    da_tmean = (da_ta850 + da_ta700 + da_ta500 + da_ta300)/4
    return da_tmean
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### OPEN/READ files
dir = Path('/home/jtibay/CORDEX3_TC/dynamics/')

dsexp43 = xr.open_dataset(dir / 'nc/merged_files/MERGED_exp43_6hourly.nc')

## ERA5
era5 = xr.open_dataset(dir / 'nc/merged_files/ERA5_20112015_remapbil.nc')
ref = era5.drop(['lat', 'lon'])
ref = era5.assign_coords({'lon': dsexp43.lon, 'lat': dsexp43.lat})
#TEMP
ref_tmean = calculate_tmean(ref)

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

bias_tmeans = []
scorr_tmeans = []

from tqdm import tqdm
for exp in tqdm(exps):
    print(f'calculating {exp} : TEMP')
    ds = xr.open_dataset(dir / f'nc/merged_files/MERGED_{exp}_6hourly.nc')
    #TMEAN
    model_tmean = calculate_tmean(ds)
    bias_tmean = model_tmean - ref_tmean
    bias_tmeans.append(bias_tmean)
    scorr = xr.corr(model_tmean, ref_tmean, dim=['lon', 'lat'])
    cc_val = "{:.2f}".format(scorr.values)
    scorr_tmeans.append(cc_val)
#----------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### set figure parameters
mp.rcParams['figure.dpi'] = 300
mp.rcParams['savefig.dpi'] = 300
mp.rcParams['font.size'] = 10
mp.rcParams['axes.linewidth'] = 1
mp.rcParams['xtick.labelsize'] = 8
mp.rcParams['ytick.labelsize'] = 8

TEMPbias_colors = ["#04125d","#163976", "#2e6292", "#6695b5", "#aec8d8",
                   "#ffffff",
                   "#dabaa3", "#c38b67", "#ab5f33", "#7b2b14", "#510b0d"]
cmapTEMP = mp.colors.ListedColormap(TEMPbias_colors, 11)
vmin = -2
vmax = 2
normTEMP = mcolors.Normalize(vmin=vmin, vmax=vmax)

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
for exp, scorrData, biasData, ax, lList in zip(exps, scorr_tmeans, bias_tmeans, 
                                               axList, letters):
    print(f'Plotting for {exp}')
    plotDA(biasData, ax, 
           f'{lList}) {exp} - ERA5 : Mean Temp', 
           f'CC:{scorrData}', 
           normTEMP, cmap=cmapTEMP, 
           transform=ccrs.PlateCarree())
cbar_ax = fig.add_axes([0.775, .18, .21, .02])
plotCBAR(cbar_ax, 
         'Mean Temperature\nat 4 levels\nDifference [\N{DEGREE SIGN}C]', 
         normTEMP, cmap=cmapTEMP)
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### save Figs
fig.savefig(dir / 'img/Fig09_tmean_diff.png',
            transparent=True,
            bbox_inches='tight')
#---------------------------------------------------------------------------------