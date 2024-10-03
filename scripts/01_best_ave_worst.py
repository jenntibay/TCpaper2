"""
Plot 5-year diffference of Best, Average, Worst experiments from ERA5:
a. MSLP
b. 850-hPa Wind Speed
c. 850-hPa Relative Vorticity
d. 300 - 850-hPa Wind Shear
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
    ax.set_title(title, fontsize=10)
    ax.text(0.83, 0.9, cc, 
            fontsize=8, fontweight='semibold',
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.grid(ls='--', alpha=0.3)
    ax.pcolormesh(da.lon, da.lat, da, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    plot_background(ax)
    ax.add_patch(plt.Rectangle((108,2), 37, 25.5, ls='--', facecolor='none', edgecolor='black', lw=1))
    return ax

def plotCBAR(axs, label, norm, cmap='jet'):
    cb1= fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), 
                  ax=axs, 
                  orientation='vertical', 
                  extend='both',
                  aspect=30,)
    cb1.ax.tick_params(labelsize=10)
    cb1.set_label(label, fontsize=10)
  
def calculate_winds(da):
    da_ua850 = da['ua'].sel(plev=850).mean(dim='time')
    da_va850 = da['va'].sel(plev=850).mean(dim='time')
    da_wspd = np.sqrt(da_ua850**2 + da_va850**2)
    return da_wspd

def calculate_relvort(da):
    da_vort = mpcalc.vorticity(da['ua'].sel(plev=850), da['va'].sel(plev=850))
    da_vort = da_vort.mean(dim='time')
    da_vort = np.multiply(da_vort, 100000)
    return da_vort

def calculate_whsear(da):
    da_ua850 = da['ua'].sel(plev=850).mean(dim='time')
    da_va850 = da['va'].sel(plev=850).mean(dim='time')
    da_ua300 = da['ua'].sel(plev=300).mean(dim='time')
    da_va300 = da['va'].sel(plev=300).mean(dim='time')
    da_ushear = da_ua300 - da_ua850
    da_vshear = da_va300 - da_va850
    da_wshear = np.sqrt(da_ushear**2 + da_vshear**2)
    return da_wshear

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
#MSLP
ref_mslp = ref.mslp.mean(dim='time')
#WIND
ref_wspd = calculate_winds(ref)
#VORT
ref_vort = calculate_relvort(ref)
#WHSEAR
ref_wshear = calculate_whsear(ref)
#TEMP
ref_tmean = calculate_tmean(ref)


## EXPs
exps = ['exp43', #Best (Top1)
        'exp06', #Average (Top19)
        'exp11', #Worst (Top39)
        ]

bias_mslps = []
scorr_mslps = []

bias_winds = []
scorr_winds = []

bias_vorts = []
scorr_vorts = []

bias_wshears = []
scorr_wshears = []

bias_tmeans = []
scorr_tmeans = []

from tqdm import tqdm
for exp in tqdm(exps):
    ds = xr.open_dataset(dir / f'nc/merged_files/MERGED_{exp}_6hourly.nc')
    #MSLP
    print(f'calculating {exp} : MSLP')
    model_mslp = ds.mslp.mean(dim='time')
    bias_mslp = model_mslp - ref_mslp
    scorr = xr.corr(model_mslp, ref_mslp, dim=['lon', 'lat'])
    cc_val = "{:.2f}".format(scorr.values)
    scorr_mslps.append(cc_val)
    bias_mslps.append(bias_mslp)
    #WIND
    print(f'calculating {exp} : WIND')
    model_wspd = calculate_winds(ds)
    bias_wind = model_wspd - ref_wspd
    bias_winds.append(bias_wind)
    scorr = xr.corr(model_wspd, ref_wspd, dim=['lon', 'lat'])
    cc_val = "{:.2f}".format(scorr.values)
    scorr_winds.append(cc_val)
    #VORT
    print(f'calculating {exp} : VORT')
    model_vort = calculate_relvort(ds)
    bias_vort = model_vort - ref_vort
    bias_vorts.append(bias_vort)
    scorr = xr.corr(model_vort, ref_vort, dim=['lon', 'lat'])
    cc_val = "{:.2f}".format(scorr.values)
    scorr_vorts.append(cc_val)
    #WSHEAR
    print(f'calculating {exp} : WSHEAR')
    model_wshear= calculate_whsear(ds)
    bias_wshear = model_wshear - ref_wshear
    bias_wshears.append(bias_wshear)
    scorr = xr.corr(model_wshear, ref_wshear, dim=['lon', 'lat'])
    cc_val = "{:.2f}".format(scorr.values)
    scorr_wshears.append(cc_val)
    #TEMP
    print(f'calculating {exp} : TEMP')
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

MSLPbias_colors = ["#2d004bff", "#5c3c82ff", "#927dacff", "#c9bed5ff", "#e4deeaff",
                   "#ffffffff",
                   "#f2e2deff", "#e5c4bdff", "#cc8a7aff", "#b24f38ff", "#a82e2aff"]
cmapMSLP = mp.colors.ListedColormap(MSLPbias_colors, 11)
vmin = -5
vmax = 5
normMSLP = mcolors.Normalize(vmin=vmin, vmax=vmax)

WINDbias_colors = ["#579ca0", "#70bcbd", "#89c6c7", "#a4d0d1", '#c0dadb',
                   '#ffffffff',
                   "#debeaf", "#d59a9f", "#c86c8a", "#b82663", "#981d52"]
cmapWIND = mp.colors.ListedColormap(WINDbias_colors, 11)
vmin = -6
vmax = 6
normWIND = mcolors.Normalize(vmin=vmin, vmax=vmax)

VORTbias_colors = ["#2d004bff", "#542788ff", "#8073acff", "#b2abd2ff",  "#d8daebff",
                   "#f7f7f7ff",
                   "#fee0b6ff","#fdb863ff", "#e08214ff", "#b35806ff", "#7f3b08ff",]
cmapVORT = mp.colors.ListedColormap(VORTbias_colors, 11)
vmin = -3
vmax = 3
normVORT = mcolors.Normalize(vmin=vmin, vmax=vmax)

WSHEARbias_colors = ["#2d004bff", "#542788ff", "#8073acff", "#b2abd2ff",  "#d8daebff",
                    "#f7f7f7ff",
                    "#fee0b6ff","#fdb863ff", "#e08214ff", "#b35806ff", "#7f3b08ff",]
cmapWSHEAR = mp.colors.ListedColormap(WSHEARbias_colors, 11)
vmin = -6
vmax = 6
normWSHEAR = mcolors.Normalize(vmin=vmin, vmax=vmax)

TEMPbias_colors = ["#04125d","#163976", "#2e6292", "#6695b5", "#aec8d8",
                   "#ffffff",
                   "#dabaa3", "#c38b67", "#ab5f33", "#7b2b14", "#510b0d"]
cmapTEMP = mp.colors.ListedColormap(TEMPbias_colors, 11)
vmin = -2
vmax = 2
normTEMP = mcolors.Normalize(vmin=vmin, vmax=vmax)
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### Plot difference of EXPs with ERA5

mosaic = """ABC
            DEF
            GHI
            JKL
            MNO"""

fig, axs = plt.subplot_mosaic(mosaic, figsize=(12, 15),
                              subplot_kw={'projection': ccrs.PlateCarree()},
                              layout='constrained')

#MSLP
axList = [axs['A'], axs['B'], axs['C']]
letters = ['a', 'f', 'k']
descriptions = ['BEST', 'AVERAGE', 'WORST']
for exp, scorrData, biasData, ax, lList, des in zip(exps, scorr_mslps, bias_mslps, 
                                               axList, letters, descriptions):
    print(f'Plotting for {exp}')
    plotDA(biasData, ax, f'{des}\n{lList}) {exp} - ERA5 : MSLP', 
           f'CC:{scorrData}', 
           normMSLP, cmap=cmapMSLP, 
           transform=ccrs.PlateCarree())
plotCBAR(axs['C'], 'MSLP Difference [hPa]', normMSLP, cmap=cmapMSLP)

#WIND
axList = [axs['D'], axs['E'], axs['F']]
letters = ['b', 'g', 'c']
for exp, scorrData, biasData, ax, lList in zip(exps, scorr_winds, bias_winds, 
                                               axList, letters):
    print(f'Plotting for {exp}')
    plotDA(biasData, ax, f'{lList}) {exp} - ERA5 : 850-hPa Wind Speed', 
           f'CC:{scorrData}', 
           normWIND, cmap=cmapWIND, 
           transform=ccrs.PlateCarree())
plotCBAR(axs['F'], 'Wind Speed Difference [m/s]', normWIND, cmap=cmapWIND)

#VORT
axList = [axs['G'], axs['H'], axs['I']]
letters = ['c', 'h', 'm']
for exp, scorrData, biasData, ax, lList in zip(exps, scorr_vorts, bias_vorts, 
                                               axList, letters):
    print(f'Plotting for {exp}')
    plotDA(biasData, ax, f'{lList}) {exp} - ERA5 : 850-hPa Rel Vort', 
           f'CC:{scorrData}', 
           normVORT, cmap=cmapVORT, 
           transform=ccrs.PlateCarree())
plotCBAR(axs['I'], 'Rel Vort Difference [$\mathregular{10^{-5}}$/s]', 
         normVORT, cmap=cmapVORT)

#WSHEAR
axList = [axs['J'], axs['K'], axs['L']]
letters = ['d', 'i', 'n']
for exp, scorrData, biasData, ax, lList in zip(exps, scorr_wshears, bias_wshears, 
                                               axList, letters):
    print(f'Plotting for {exp}')
    plotDA(biasData, ax, f'{lList}) {exp} - ERA5 : 300-850-hPa Wind Shear ', 
           f'CC:{scorrData}', 
           normWSHEAR, cmap=cmapWSHEAR, 
           transform=ccrs.PlateCarree())
plotCBAR(axs['L'], 'Wind Shear Difference [m/s]', 
         normWSHEAR, cmap=cmapWSHEAR)

#WSHEAR
axList = [axs['M'], axs['N'], axs['O']]
letters = ['e', 'j', 'o']
for exp, scorrData, biasData, ax, lList in zip(exps, scorr_tmeans, bias_tmeans, 
                                               axList, letters):
    print(f'Plotting for {exp}')
    plotDA(biasData, ax, f'{lList}) {exp} - ERA5 : Mean Temperature ', 
           f'CC:{scorrData}', 
           normTEMP, cmap=cmapTEMP, 
           transform=ccrs.PlateCarree())
plotCBAR(axs['O'], 'Mean Temp Difference [\N{DEGREE SIGN}C]', 
         normTEMP, cmap=cmapTEMP)
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### save Fig
fig.savefig(dir / 'img/Fig10_best_ave_worst_diff.png',
            transparent=True,
            bbox_inches='tight')
#---------------------------------------------------------------------------------
