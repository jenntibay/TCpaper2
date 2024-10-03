"""
Plot Heatmap of 1-NRMSE, 1-NMBE, CC of each experiment and its RANK

Created on 02 Oct 2024
by jtlopez
"""
#---------------------------------------------------------------------------------
#%%
from pathlib import Path
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from pylab import rcParams
import seaborn as sns
#%%
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### Define functions

def plotHEATMAP(df, ax, title, weight, norm, fmt='.2f', cmap='jet'):
    g1 = sns.heatmap(df, norm=norm,
                     annot=True,
                     annot_kws={"fontsize":8, "alpha":0.8},
                     fmt=".2f",
                     cmap=cmap,
                     linewidth=.5,
                     ax=ax,
                     cbar=False,
                     )
    g1.set_ylabel('')
    ax.set_title(title, weight='bold')
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    ax.tick_params(axis='y', labelrotation=0)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        horizontalalignment='center',
        weight=weight)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, 
                                 fc='none', ec='black',
                                 clip_on=False, 
                                 transform=ax.transAxes))

def plotHEATMAP_RANK(df, ax, title, weight, norm, cmap='jet'):
    g1 = sns.heatmap(df, norm=norm,
                     annot=True,
                     annot_kws={"fontsize":11, "alpha":1},
                     cmap=cmap,
                     linewidth=.5,
                     ax=ax,
                     cbar=False,
                     )
    g1.set_ylabel('')
    ax.set_title(title, weight='bold')
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    ax.tick_params(axis='y', labelrotation=0)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        horizontalalignment='center',
        weight=weight)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, 
                                 fc='none', ec='black',
                                 clip_on=False, 
                                 transform=ax.transAxes))

def plotCBAR(ax, label, norm, cmap='jet'):
    cb1= fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), 
                  ax=ax, 
                  orientation='horizontal',
                  pad=0.025)
    cb1.ax.tick_params(labelsize=10)
    cb1.set_label(label, fontsize=10)

def plotCBAR_RANK(ax, label, ticks, norm, cmap='jet'):
    cb1= fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), 
                  ax=ax, 
                  orientation='horizontal',
                  pad=0.025)
    cb1.ax.tick_params(labelsize=10)
    cb1.set_label(label, fontsize=10)    
    cb1.set_ticks(ticks)
    cb1.ax.set_xticklabels(ticks)
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### OPEN/READ files
dir = Path('/home/jtibay/CORDEX3_TC/heatmap/')

df_rmse = pd.read_csv(dir / 'data/score_rmse.csv', index_col='METRICS')
df_mbe = pd.read_csv(dir / 'data/score_mbe.csv', index_col='METRICS')
df_scorr = pd.read_csv(dir / 'data/score_scorr.csv', index_col='METRICS')
df_opei = pd.read_csv(dir / 'data/score_opei.csv', index_col='METRICS')
dr = pd.read_csv(dir / 'data/rank.csv', index_col='METRICS')
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### set figure parameters
mp.rcParams['figure.dpi'] = 300
mp.rcParams['savefig.dpi'] = 300
mp.rcParams['font.size'] = 10
mp.rcParams['axes.linewidth'] = 1
mp.rcParams['xtick.labelsize'] = 10
mp.rcParams['ytick.labelsize'] = 10

cm1_data = np.loadtxt("/home/jtibay/ScientificColourMaps7/lapaz/lapaz.txt")
cm.register_cmap(cmap=mp.colors.LinearSegmentedColormap.from_list("lapaz", cm1_data, 10).reversed())
#lapaz_map = mp.colors.LinearSegmentedColormap.from_list('', cm1_data, 10)

vmin=0
vmax=1
normSCORE = mcolors.Normalize(vmin=vmin, vmax=vmax)

rank_colors = ["#420239ff", "#5e0146ff", "#872351ff", "#9b3557ff", "#d86868ff", 
               "#df817aff", "#ecb29eff", "#f2cbb0ff", "#fffcd4ff"]
cmapRANK = mp.colors.ListedColormap(rank_colors, 9)
vmin=1
vmax=40
normRANK = mcolors.Normalize(vmin=vmin, vmax=vmax)
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### Plot Heatmap

mosaic = """ABCDE"""

fig, axs = plt.subplot_mosaic(mosaic, figsize=(14, 14),
                              sharey=True,
                              gridspec_kw={'width_ratios':[1,1,0.07,0.07,0.07],
                                         'wspace':.1},
                              layout='constrained'
                              )

#Plot heatmap for 1-NRMSE, 1-NMBE, CC-TCTD, OPEI
dfs = [df_rmse, df_mbe, df_scorr, df_opei]
axList = [axs['A'], axs['B'], axs['C'], axs['D']]
titles = ['1-NRMSE', '1-NMBE', '', '',]
weights = ['normal', 'normal', 'bold', 'bold']
for df, ax, title, weight in zip(dfs, axList, titles, weights):
    plotHEATMAP(df, ax, f'{title}', weight, norm=normSCORE, fmt='.2f', cmap='lapaz_r')
plotCBAR(axs['A'], 'Score', norm=normSCORE, cmap='lapaz_r')

#Plot heatmap for RANK
plotHEATMAP_RANK(dr, axs['E'], '', 'bold', norm=normRANK, cmap=cmapRANK)
ticks = [1,5,10,15,20,25,30,35,40]
plotCBAR_RANK(axs['B'], 'Rank', ticks, norm=normRANK, cmap=cmapRANK)
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### save Fig
fig.savefig('/home/jtibay/CORDEX3_TC/dynamics/img/Fig04_heatmap.png',
            transparent=True,
            bbox_inches='tight')
#---------------------------------------------------------------------------------