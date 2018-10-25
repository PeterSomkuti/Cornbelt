import h5py
import numpy as np
import pandas as pd
from netCDF4 import num2date

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times New Roman']

from matplotlib import pyplot as plt
import palettable as pt
import statsmodels.formula.api as smf

###############################################################################
## Change analysis parameters here
###############################################################################

resample = 'MS'
rs_apply = 'mean'
min_count = 50
min_date = '20100101'
max_date = '20161231'
anomaly_stat = 'median'

###############################################################################


######### PREPPING DATA ###########

### Read in the USDA data

# Planted area
df_area_corn = pd.read_csv('area_planted_all_corn.csv', sep=',', thousands=',')
df_area_soy = pd.read_csv('area_planted_all_soy.csv', sep=',', thousands=',')
df_area_wheat = pd.read_csv('area_planted_all_wheat.csv', sep=',', thousands=',')

planted_area_corn = df_area_corn.query('Period == "YEAR"').groupby('Year')['Value'].sum()
planted_area_soy = df_area_soy.query('Period == "YEAR"').groupby('Year')['Value'].sum()
planted_area_wheat = df_area_wheat.query('Period == "YEAR"').groupby('Year')['Value'].sum()

all_planted_area = planted_area_corn + planted_area_soy + planted_area_wheat
all_planted_area_rel = all_planted_area / all_planted_area[2010]
# Production

usda_production = pd.read_csv('corn_soy_wheat_production.csv', sep=',', thousands=',').query('Program == "SURVEY"')


# Read HDF file, extract and convert dates
h5 = h5py.File('fluorescence_subset.h5', 'r')
dates = num2date(h5['frac_days_since'][:], units='days since 1970-01-01')

qual = (h5['qual_flag'][:] == 1)

# Create dataframe using the dates, and then read all 1-dim data arrays into it
df = pd.DataFrame(index=dates[qual])

for key in h5.keys():
    if len(h5[key].shape) == 1:
        if h5[key].shape[0] == len(h5['Exposure_id']):
            # Don't want the empty 'simple' fields
            if 'simple' in key: continue
            
            print(f"Reading {key}")
            df[key] = h5[key][:][qual]

# Here are the keys for the SIF data fields
abs_755_list = [x for x in df.columns if
                  (('corr' in x) and ('abs' in x) and ('755' in x))]
abs_772_list = [x for x in df.columns if
                  (('corr' in x) and ('abs' in x) and ('772' in x))]
abs_755_list_P = [x for x in df.columns if
                  (('corr' in x) and ('abs' in x) and ('755_P' in x))]
abs_755_list_S = [x for x in df.columns if
                  (('corr' in x) and ('abs' in x) and ('755_S' in x))]
abs_772_list_P = [x for x in df.columns if
                  (('corr' in x) and ('abs' in x) and ('772_P' in x))]
abs_772_list_S = [x for x in df.columns if
                  (('corr' in x) and ('abs' in x) and ('772_S' in x))]

# Scale SIF to units of W/m2/sr/nm
df.loc[:, abs_755_list] *= (1e7 / 755)**2
df.loc[:, abs_772_list] *= (1e7 / 772)**2

# Get CASA fluxes from gC/m2/s to gC/m2/yr
df.loc[:, ['CASA_GEE', 'CASA_NEE']] *= 60*60*24*365

# Set bad SM values to NaNs
sm_list = ['sm_combined', 'sm_active', 'sm_passive']
for sm in sm_list:
    neg_sm = df.loc[:, sm] < 0
    df.loc[neg_sm, sm] = np.nan 



# Make Gome-2 use only overlapping data
g2_ov = df.loc[:, 'gome2_overlap'] < 0.9
df.loc[g2_ov, 'gome2_sif'] = np.nan

# Resample the data according to resample period
dfs = df[min_date:max_date].resample(resample).apply(rs_apply)
# Save how many soundings we have
dfs['count'] = df[min_date:max_date].loc[:, 'Exposure_id'].resample(resample).count().values
# And drop all rows which are below the minimum count threshold
dfs.loc[dfs['count'] < min_count, :] = np.nan


# Normalise sif to planted area?
for year in np.unique(dfs.index.year):
    year_mask = dfs.index.year == year
    dfs.loc[year_mask, abs_755_list] *= all_planted_area_rel[year]


# Create anomaly dataframe
dfa = dfs.copy()

# Anomalies for monthlies
for m, dfg in dfa.groupby(dfa.index.month):
    dfa[dfa.index.month == m] -= dfg.apply(anomaly_stat)

# GRACE is already an anomaly, so we put that data back in
dfa.loc[:, 'GRACE'][:] = dfs.loc[:, 'GRACE'][:]
dfa.loc[:, 'count'][:] = dfs.loc[:, 'count'][:]

##### Make a nice overview plot, showing the SIF anomalies in the context of
##### other anomalies taken from the dataset
def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)
    
def ts_overview(plot_df, sif_list, aux_list,
                aux_labels=None,
                fname='ts_grid.png',
                ncol=1, hline=0.0, symm_lim=False):

    nrow = int(np.ceil((len(aux_list) + 2 )/ ncol))

    fig, axarr = plt.subplots(nrow, ncol, figsize=(8, 8), dpi=300)
    axlist = axarr.flatten()
    
    # Make SIF time series as first entry
    sifcolor='g'
    ax = axlist[0]
    sif_data = plot_df.loc[:, sif_list]
    sif_lim = np.max(np.abs(sif_data.median(axis=1))) * 1.3
    # Plot median of all SIF data
    ax.plot(sif_data.median(axis=1), color=sifcolor,
            label='SIF [W/m$^2$/sr/nm]')
    ax.legend(fontsize=7, loc='lower left', framealpha=1.0)
    # Standard deviation to spot where significant anomalies lie
    if hline is not None:
        ax.hlines([-sif_data.median(axis=1).std(),
                   sif_data.median(axis=1).std()],
                  xmin=sif_data.index[0], xmax=sif_data.index[-1],
                  linestyle='dashed', color='g', lw=1.0)

    # Fill area between min and max of all SIF data
    ax.fill_between(x=plot_df.index,
                    y1=sif_data.loc[:, sif_list].min(axis=1),
                    y2=sif_data.loc[:, sif_list].max(axis=1),
                    color=sifcolor,
                    alpha=0.25)
    if symm_lim:
        ax.set_ylim(-sif_lim, sif_lim)
    if hline != None:
        ax.hlines(hline, plot_df.index[0], plot_df.index[-1],
                  colors='grey', linestyles='dashed',
                  linewidth=0.5)
    ax.set_xticklabels([])

    # Plot the number of measurements
    ax = axlist[1]
    ax.plot(plot_df['count'], label='GOSAT count', color='k')
    ax.legend(fontsize=7, loc='lower left', framealpha=1.0)
    ax.set_xticklabels([])

    # Then fill the rest in with other data
    aux_colors = pt.tableau.BlueRed_12.mpl_colors
    for i, aux in enumerate(aux_list):

        r = plot_df.loc[:, aux].corr(plot_df.loc[:, sif_list].median(axis=1))

        aux_lim = np.max(np.abs(plot_df.loc[:, aux])) * 1.3
        
        ax = axlist[i+2]

        if aux_labels == None:
            aux_label = aux
        else:
            aux_label = aux_labels[i]
        
        ax.plot(plot_df.loc[:, aux],
                #color=aux_colors[i],
                label=aux_label)
        if symm_lim:
            ax.set_ylim(-aux_lim, aux_lim)
        #ax.set_title(aux)
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        
        if hline != None:
            ax.hlines(hline, plot_df.index[0], plot_df.index[-1],
                      colors='grey', linestyles='dashed', linewidth=0.5)
        ax2 = ax.twinx()
        ax2.plot(sif_data.median(axis=1),
                 color=sifcolor, linestyle='dashed', lw=1.0,
                 zorder=5)
        ax2.set_yticklabels([])
        ax2.legend(ax_handles, ax_labels,
                   fontsize=7, loc='lower left', framealpha=1.0)
        ax2.text(0.98, 0.08,
                 'r = {:.3f}'.format(r),
                 ha='right', va='bottom',
                 fontsize=7, transform=ax.transAxes,
                 fontdict={'family': 'monospace'},
                 bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'),
                 zorder=999)
        
        if symm_lim:
            ax2.set_ylim(-sif_lim, sif_lim)    
            align_yaxis(ax, 0, ax2, 0)
        
        if i < len(aux_list)-ncol:
            ax.set_xticklabels([])

    fig.tight_layout(h_pad=1, w_pad=2)
    fig.align_ylabels()
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    
###############################################################################



# Data prep finished - now make some nice plots and calculations


aux_list = ['gome2_sif', 'myd13_evi',
            #'albedo_1',
            'trmm_precip', 'GRACE',
            'myd11_lst_day', 'sm_combined',
            'fvc', 'lai', 'CASA_GEE', 'CASA_NEE']
aux_labels = ['GOME-2 SIF [W/m$^2$/sr/nm]',
              'EVI',
              #'Albedo 755nm',
              'TRMM [mm/day]',
              'GRACE [?]',
              'LST [K]',
              'Combined SM [m$^3$/m$^3$]',
              'FVC [?]',
              'LAI',
              'CASA GPP [gC/m$^2$/yr]',
              'CASA NPP [gC/m$^2$/yr]']

ts_overview(dfa, abs_755_list_P, aux_list,
            aux_labels=aux_labels, ncol=2,
            symm_lim=True, fname='anomalies_755_abs_P.png')

ts_overview(dfa, abs_755_list_S, aux_list,
            aux_labels=aux_labels, ncol=2,
            symm_lim=True, fname='anomalies_755_abs_S.png')
"""


ts_overview(dfs, abs_755_list, aux_list,
            aux_labels=aux_labels, ncol=2, hline=None,
            fname='time_series_755_abs.png')
ts_overview(dfa, abs_755_list, aux_list,
            aux_labels=aux_labels, ncol=2,
            symm_lim=True, fname='anomalies_755_abs.png')
ts_overview(dfa, rel_755_list, aux_list,
            aux_labels=aux_labels, ncol=2,
            symm_lim=True, fname='anomalies_755_rel.png')

ts_overview(dfs, abs_772_list, aux_list, ncol=2, hline=None,
            fname='time_series_772_abs.png')
ts_overview(dfa, abs_772_list, aux_list, ncol=2,
            symm_lim=True, fname='anomalies_772_abs.png')
ts_overview(dfa, rel_772_list, aux_list, ncol=2,
            symm_lim=True, fname='anomalies_772_rel.png')
"""

# Build a new DF for model analysis
dfm = pd.DataFrame(index=dfs.index,
                   columns=['sif',] + aux_list)
dfm['sif'] = dfa.loc[:, abs_755_list].median(axis=1)
for aux in aux_list:
    dfm.loc[:, aux] = dfa.loc[:, aux].values


