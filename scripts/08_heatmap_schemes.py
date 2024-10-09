"""
Plot Heatmap of Experiments schemes

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
### OPEN/READ files
dir = Path('/home/jtibay/CORDEX3_TC/heatmap/')

dr = pd.read_csv(dir / 'data/rank.csv', index_col='METRICS')
df_rank = dr.sort_values(by='RANK', ascending=True)
df_schemes = pd.read_csv(dir / 'data/regcm4NH_exps_schemes.csv', index_col='EXP')

for index in df_rank.index:
    df_rank.loc[index, 'CP'] = df_schemes.loc[index, 'CP']
    df_rank.loc[index, 'PBL'] = df_schemes.loc[index, 'PBL']
    df_rank.loc[index, 'MP'] = df_schemes.loc[index, 'MP']

df_rank['CP'] = df_rank['CP'].str.replace('Kain-Fritsch','KF')
df_rank['CP'] = df_rank['CP'].str.replace('Grell','GR')
df_rank['CP'] = df_rank['CP'].str.replace('Tiedtke','TE')
df_rank['PBL'] = df_rank['PBL'].str.replace('Holtslag','H')
df_rank['PBL'] = df_rank['PBL'].str.replace('GFS2011','GFS')

df_rank = df_rank.drop(columns=['RANK'])

#CP: 0,1,2,3: GR, KF, TE, MIT
#PBL: 0,1,2,3:GFS, UW, H, MYJ
#MP: 0,1,2: SUB, WSM5, NT
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### set figure parameters
mp.rcParams['figure.dpi'] = 300
mp.rcParams['savefig.dpi'] = 300
mp.rcParams['font.size'] = 10
mp.rcParams['axes.linewidth'] = 1
mp.rcParams['xtick.labelsize'] = 10
mp.rcParams['ytick.labelsize'] = 10

cp_colors = ["#262626", "#2E5F7F", "#64B1DD", "#93CAE8", '#C7E5F3']
cmapCP = mp.colors.ListedColormap(cp_colors, 5)

pbl_colors = ["#262626", "#93358F", "#CA73C8", "#D9D9D9", "#A6A6A6"]
cmapPBL = mp.colors.ListedColormap(pbl_colors, 5)

mp_colors = ["#262626", "#F5C242", "#F0E9AB", "#D0D0D0"]
cmapMP = mp.colors.ListedColormap(mp_colors, 4)

cmap_dict = {
    'CP': cmapCP,
    'PBL': cmapPBL,
    'MP': cmapMP,
}
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### Plot Heatmap
mosaic = """ABC"""

fig, axs = plt.subplot_mosaic(mosaic, figsize=(3,10), 
                              sharey=True,
                              gridspec_kw={'wspace':0},
                              layout='constrained')

axList=[axs['A'], axs['B'], axs['C']]
for col, ax in zip(df_rank.columns, axList):
    # Create a numeric representation for coloring
    heatmap_data = df_rank[[col]].copy()
    heatmap_data[col] = pd.factorize(heatmap_data[col])[0]
    # Create the heatmap
    g1 = sns.heatmap(heatmap_data, ax=ax,
                     cmap=cmap_dict[col], 
                     cbar=False, 
                     annot=df_rank[[col]], 
                     fmt='', 
                     linewidths=0.5)
    g1.set_ylabel('')
    ax.set_title(col, weight='bold')
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    ax.tick_params(axis='y', labelrotation=0)
    ax.set_xticklabels([])
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
### save Fig
fig.savefig('/home/jtibay/CORDEX3_TC/dynamics/img/SFig01_heatmap_schemes.png',
            transparent=True,
            bbox_inches='tight')
#---------------------------------------------------------------------------------
