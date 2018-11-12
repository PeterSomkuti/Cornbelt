import matplotlib
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times', 'Times New Roman']

rcParams['xtick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7
rcParams['axes.labelsize'] = 8
rcParams['mathtext.fontset'] = 'stix'

from matplotlib import pyplot as plt

import h5py
import numpy as np
import pandas as pd
from netCDF4 import num2date, Dataset

import scipy as sp
from scipy import stats as sps
from scipy.interpolate import RegularGridInterpolator as RGI
import itertools

import palettable as pt
import statsmodels.formula.api as smf

from bbox import *

###############################################################################
## Change analysis parameters here
###############################################################################

resample = 'MS'
rs_apply = 'mean'
min_sif = 0.0
min_count = 200
min_date = '20100101'
max_date = '20161231'
anomaly_stat = 'median'

problem = ['2013-04-01', '2014-02-01', '2014-03-01', '2014-04-01',
           '2014-11-01', '2015-02-01', '2015-04-01', '2016-08-01']

###############################################################################


def area(lat1, lat2, lon1, lon2, R=6371000):
    # Radius in meters, so we get area in m^2

    lat1x, lat2x, lon1x, lon2x = np.deg2rad([lat1, lat2, lon1, lon2])
    result = R**2 * (np.sin(lat1x) - np.sin(lat2x)) * (lon1x - lon2x)
    return np.abs(result)

######### PREPPING DATA #######################################################


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

# Progress
df_progress_corn = pd.read_csv('progress_corn.csv', sep=',', thousands=',')
df_progress_corn.index = pd.to_datetime(df_progress_corn['Week Ending']).values
df_progress_corn['WeekNr'] = [int(x[-2:]) for x in df_progress_corn['Period']]

df_progress_soy = pd.read_csv('progress_soy.csv', sep=',', thousands=',')
df_progress_soy.index = pd.to_datetime(df_progress_soy['Week Ending']).values
df_progress_soy['WeekNr'] = [int(x[-2:]) for x in df_progress_soy['Period']]

# Production and Yield
df_production = pd.read_csv('corn_soy_production.csv', sep=',', thousands=',')
df_yield = pd.read_csv('corn_soy_yield.csv', sep=',', thousands=',')


# Read HDF file, extract and convert dates
h5 = h5py.File('fluorescence_subset.h5', 'r')
dates = num2date(h5['frac_days_since'][:], units='days since 1970-01-01')

# Create dataframe using the dates, and then read all 1-dim data arrays into it
df = pd.DataFrame(index=dates)

print("Reading HDF")
for key in h5.keys():
    if len(h5[key].shape) == 1:
        if h5[key].shape[0] == len(h5['Exposure_id']):
            # Don't want the empty 'simple' fields
            if 'simple' in key: continue
            
            #print(f"Reading {key}")
            df[key] = h5[key][:] #[qual]
print("Done")
# Here are the keys for the SIF data fields

SL = dict() # Sif key list
for absrel, pol, window in itertools.product(
        ['abs', 'rel'], ['P', 'S', ''], ['755', '772']):
    if pol == '':
        polstr = ''
    else:
        polstr = f'_{pol}'

    dkey = f"{absrel}_{window}{polstr}"

    SL[dkey] = [x for x in df.columns if
                (('corr' in x) and (absrel in x) and (f'{window}{polstr}' in x))]


# Scale SIF to units of W/m2/sr/nm
df.loc[:, SL['abs_755']] *= (1e7 / 755)**2
df.loc[:, SL['abs_772']] *= (1e7 / 772)**2

# DPsurf filter? Does not seem to do anything significant
#dpfilt = (df['r_svsv_psurf_772'] - df['r_svap_psurf_772']) < -10000
#df.loc[dpfilt, :] = np.nan

# Get CASA fluxes from kgC/m2/s to gC/m2/yr
df.loc[:, ['CASA_GEE', 'CASA_NEE']] *= 60*60*24*365*1000

# Set bad SM values to NaNs
sm_list = ['sm_combined', 'sm_active', 'sm_passive']
for sm in sm_list:
    bad_sm = df.loc[:, sm+'_uncrt'] <= 0
    df.loc[bad_sm, sm] = np.nan 



# Make Gome-2 use only overlapping data
g2_ov = df.loc[:, 'gome2_overlap'] < 0.9
df.loc[g2_ov, 'gome2_sif'] = np.nan

# Read UoE fluxes and put in dataframe
df['uoe-7.1b'] = np.nan

nc = Dataset('u2ol_7.1b_fluxmap_uoe_r2012_1x1.nc', 'r')
#nc = Dataset('u72ol_fluxmap_uoe_1x1.nc', 'r')

uoe_lats = nc.variables['lat'][:]
uoe_lons = nc.variables['lon'][:]

date_array = nc.variables['start_date'][:]
dates_uoe = pd.to_datetime(['{:d}-{:02d}-{:02d}'.format(x[0], x[1], x[2])
                            for x in date_array])

lon1_idx = np.argmin(np.abs(uoe_lons - lon_min))
lon2_idx = np.argmin(np.abs(uoe_lons - lon_max))
lat1_idx = np.argmin(np.abs(uoe_lats - lat_min))
lat2_idx = np.argmin(np.abs(uoe_lats - lat_max))

df_flux = pd.DataFrame(index=dates_uoe, columns=['uoe-7.1b'])

flux = nc.variables['flux'][:,:,:] # these come as kgC/s/m2
flux *= 86400 * 1e3 # this is now in gC/day

for year in range(2009, 2017):
    for month in range(1,13):
 
        df_select = ((df.index.year == year) &
                     (df.index.month == month))
        uoe_idx = np.where((dates_uoe.year == year) &
                           (dates_uoe.month == month))[0][0]

        gridint = RGI((uoe_lats, uoe_lons), flux[uoe_idx])

        df.loc[df_select, 'uoe-7.1b'] = gridint(np.dstack([df.loc[df_select, 'lat'].values,
                                                           df.loc[df_select, 'lon'].values])[0])




# Resample the data according to resample period
dfs = df[min_date:max_date].resample(resample).apply(rs_apply)
# Save how many soundings we have
dfs['count'] = df[min_date:max_date].loc[:, 'Exposure_id'].resample(resample).count().values
# And drop all rows which are below the minimum count threshold
dfs.loc[dfs['count'] < min_count, :] = np.nan

for c in SL['abs_755']+SL['abs_772']:
    too_small = dfs.loc[:, c] < min_sif
    dfs.loc[too_small, c] = np.nan


# Normalise sif to planted area? Does not significantly change anything..
for year in np.unique(dfs.index.year):
    year_mask = dfs.index.year == year
#    dfs.loc[year_mask, abs_755_list] *= all_planted_area_rel[year]



dfs.loc[:, 'uoe-7.1b'] *= dfs.index.daysinmonth * 12
#df_flux.loc[:, 'uoe-7.1b'] = np.mean(flux, axis=(1,2)) * df_flux.index.daysinmonth # / 1e12 # TgC

#dfs['uoe-7.1b'] = df_flux[min_date:max_date].loc[:, 'uoe-7.1b']
    
# Create anomaly dataframe
dfa = dfs.copy()

# Anomalies for monthlies
for (m,d), dfg in dfa.groupby([dfa.index.month,
                              dfa.index.day]):
    dfa[(dfa.index.month == m) & (dfa.index.day == d)] -= dfg.apply(anomaly_stat)

# GRACE is already an anomaly, so we put that data back in
dfa.loc[:, 'GRACE'][:] = dfs.loc[:, 'GRACE'][:]
dfa.loc[:, 'spei'][:] = dfs.loc[:, 'spei'][:]
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
                figsize=(5, 6),
                aux_labels=None,
                plot_sif=True,
                fname='ts_grid.png',
                ncol=1, hline=0.0, symm_lim=False):

    sif_count = 0
    if plot_sif:
        sif_count = 2
        
    nrow = int(np.ceil((len(aux_list) + sif_count)/ ncol))

    fig, axarr = plt.subplots(nrow, ncol, figsize=figsize, dpi=300)
    axlist = axarr.flatten()

    sif_data = plot_df.loc[:, sif_list]
    sif_lim = np.max(np.abs(sif_data.median(axis=1))) * 1.3
    sifcolor='g'
    
    if plot_sif:
        # Make SIF time series as first entry
        ax = axlist[0]

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
        
        ax = axlist[i+sif_count]

        if aux_labels == None:
            aux_label = aux
        else:
            aux_label = aux_labels[i]
        
        ax.plot(plot_df.loc[:, aux],
                #color=aux_colors[i],
                label=aux_label, zorder=5)
        '''
        for p in problem:
            pl = np.where(plot_df.index == p)[0][0]
            ax.scatter(plot_df.index[pl],
                       plot_df.iloc[pl].loc[aux],
                       s=3, color='r', zorder=10)
        '''               
        
        if symm_lim:
            ax.set_ylim(-aux_lim, aux_lim)
            if ('GPP' in aux_label) or ('NPP' in aux_label):
                ax.invert_yaxis()

            
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

# First, plot the SIF time series, along with anomalies and measurement count
for pol in ['_P', '_S', '']:
    fig, axarr = plt.subplots(2, 1, figsize=(4, 2.5))

    sl = axarr[0].plot(dfs.loc[:, SL[f'abs_755{pol}']].median(axis=1), 'g-')
    axarr[0].fill_between(dfs.index,
                          y1=dfs.loc[:, SL[f'abs_755{pol}']].min(axis=1),
                          y2=dfs.loc[:, SL[f'abs_755{pol}']].max(axis=1),
                          color='green', alpha=0.15)

    axarr[0].set_ylabel('SIF\n[mW/m$\mathrm{^2}$/sr/nm]')
    axarr[0].set_xticklabels([])

    ax2 = axarr[0].twinx()
    gc = ax2.plot(dfs['count'], '--', linewidth=1, label='GOSAT Count')
    ax2.set_xticklabels([])
    ax2.set_ylabel('Count')

    axarr[1].plot(dfa.loc[:, SL[f'abs_755{pol}']].median(axis=1), 'g-')
    axarr[1].fill_between(dfa.index,
                          y1=dfa.loc[:, SL[f'abs_755{pol}']].min(axis=1),
                          y2=dfa.loc[:, SL[f'abs_755{pol}']].max(axis=1),
                          color='green', alpha=0.15)

    anomaly_std = dfa.loc[:, SL[f'abs_755{pol}']].median(axis=1).std()
    
    axarr[1].hlines([0], xmin=dfa.index[0], xmax=dfa.index[-1],
                    color='grey', linewidth=0.5, linestyle='solid')
    axarr[1].hlines([-anomaly_std, anomaly_std], xmin=dfa.index[0], xmax=dfa.index[-1],
                    color='green', linewidth=0.5, linestyle='dotted')
    
    axarr[1].set_ylabel('SIF Anomaly\n[mW/m$^2$/sr/nm]')

    fig.align_ylabels()

    #fig.tight_layout()
    fig.legend([sl[0], gc[0]], ['Solar-induced Fluorescence (SIF)',
                                'GOSAT Measurement Count'],
               loc='lower center', fontsize=7, bbox_to_anchor=(0.5, -0.01))

    fig.subplots_adjust(bottom=0.2)
    plt.savefig(f'SIF_overview{pol}.pdf', bbox_inches='tight')
    plt.close('all')

aux_list = ['gome2_sif',
            'GRACE',
            'trmm_precip',
            #######
            'myd13_evi',
            'fvc',
            'lai',
            #######
            'myd11_lst_day',
            'sm_combined',
            'sm_active',
            #####
            'CASA_GEE', 'CASA_NEE', 'uoe-7.1b']

aux_labels = ['GOME-2 SIF [mW/m$^2$/sr/nm]',
              'GRACE',
              'TRMM [mm/hr]',
              ####
              'EVI',
              'FVC',
              'LAI',
              ####
              'LST [K]',
              'Combined SM [m$^3$/m$^3$]',
              'Active SM [%]',
              ####
              'CASA GPP [gC/m$^2$/yr]',
              'CASA NPP [gC/m$^2$/yr]',
              'UoE NPP [gC/m$^2$/yr]']

for absrel, pol, window in itertools.product(
        ['abs', 'rel'], ['_P', '_S', ''], ['755', '772']):
    for x in range(4):
        sl = slice(x*3, (x+1)*3)
        ts_overview(dfa, SL[f'{absrel}_{window}{pol}'], aux_list[sl],
                    aux_labels=aux_labels[sl],
                    plot_sif=False, ncol=1, figsize=(4, 3.5), 
                    symm_lim=True, fname=f'anomalies_{window}_{absrel}{pol}_{x}_{anomaly_stat}.pdf')

ts_overview(dfa, SL['rel_755_P'], aux_list,
            aux_labels=aux_labels, ncol=2,
            symm_lim=True, fname='anomalies_755_rel_P.pdf')
ts_overview(dfa, SL['abs_755_P'], aux_list,
            aux_labels=aux_labels, ncol=2,
            symm_lim=True, fname='anomalies_755_abs_P.pdf')

ts_overview(dfa, SL['abs_755_S'], aux_list,
            aux_labels=aux_labels, ncol=2,
            symm_lim=True, fname='anomalies_755_abs_S.pdf')

# THIS GIVES BEST ANOMALY CORRELATIONS..
ts_overview(dfa, SL['abs_755'], aux_list,
            aux_labels=aux_labels, ncol=2,
            symm_lim=True, fname='anomalies_755_abs.pdf')

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
                   columns=['sif', 'albedo_1', 'SZA'] + aux_list)
dfm['sif'] = dfa.loc[:, SL['abs_755']].median(axis=1)
for aux in dfm.columns:
    if aux == 'sif':
        continue
    dfm.loc[:, aux] = dfa.loc[:, aux].values


# Here we analyse SIF vs crop progress

dfp_corn = df_progress_corn[df_progress_corn['Data Item'] ==
                            'CORN - PROGRESS, MEASURED IN PCT PLANTED']
dfp_soy = df_progress_soy[df_progress_soy['Data Item'] ==
                          'SOYBEANS - PROGRESS, MEASURED IN PCT PLANTED']




# Make 2-D plot of SIF integration months vs CROP WEEK
corr_max = 0.0
for sif_integral in range(2,13):
    for crop_week in range(15, 25):

        xdata = []
        ydata = []

        dfp = dfp_corn.copy()
        df_area = df_area_corn.copy()
        try:
            for year in range(2010, 2017):

                for state in np.unique(dfp['State']):
                    this_area = df_area.query(f'Year == {year} & State == "{state}" & Period == "YEAR"')
                    dfp.loc[(dfp['Year'] == year) & (dfp['State'] == state), 'Value'] *= this_area.Value.values            


                sif_data = dfs.loc[dfs.index.year == year, SL['abs_755']]
                sif_mean = sif_data.mean(axis=1)
                sif_std = sif_data.std(axis=1)
                
                crop_data = dfp[dfp.index.year == year]['Value'].resample('W').sum()
                crop_std = dfp[dfp.index.year == year]['Value'].resample('W').std()
                
                crop_idx = np.where(crop_data.index.week == crop_week)[0][0]
        
                xdata.append(sif_mean[2:sif_integral].sum())
                ydata.append(crop_data[crop_idx])

            if corr_max < sps.pearsonr(xdata, ydata)[0]:
                print(sif_integral, crop_week, sps.pearsonr(xdata, ydata)[0])
                corr_max = sps.pearsonr(xdata, ydata)[0]
        except:
            pass
            # print(sif_integral, crop_week, 'failed')


sif_integral = [5, 5] #[6, 7]
crop_week = [18, 18]
plottitle = ['Corn', 'Soybeans']
markerlist = ['o', '^', 's', '8', '<', 'p', 'D']

fig, axarr  = plt.subplots(1, 2, figsize=(4.5, 2.5), dpi=300)
for i, (dfp, df_area) in enumerate([(dfp_corn.copy(), df_area_corn.copy()),
                                    (dfp_soy.copy(), df_area_soy.copy())]):

    ax = axarr[i]
    
    xdata = []
    ydata = []
    lines = []
    for j, year in enumerate(range(2010, 2017)):

        sif_data = dfs.loc[dfs.index.year == year, SL['abs_755']]
        sif_mean = sif_data.mean(axis=1)
        sif_std = sif_data.std(axis=1)

        for state in np.unique(dfp['State']):
            this_area = df_area.query(f'Year == {year} & State == "{state}" & Period == "YEAR"')
            dfp.loc[(dfp['Year'] == year) & (dfp['State'] == state), 'Value'] *= this_area.Value.values            
        

        # We are deviding by 1e8 because million * percent
        crop_data = dfp[dfp.index.year == year]['Value'].resample('W').sum() / 1e8
        #crop_std = dfp[dfp.index.year == year]['Value'].resample('W').std() * 0.0
    
        crop_idx = np.where(crop_data.index.week == crop_week[i])[0][0]
        
        xdata.append(sif_mean[1:sif_integral[i]].sum())
        ydata.append(crop_data[crop_idx])
    
        l1, l2, l3 = ax.errorbar(x=xdata[-1],
                                 y=ydata[-1],
                                 xerr=np.sqrt(np.sum(sif_std[2:sif_integral[i]]**2)),
                                 #yerr=crop_std[crop_idx],
                                 fmt=markerlist[j], label=year,
                                 markersize=3,
                                 capsize=1.0,
                                 elinewidth=1.0)

        lines.append(l1)

    ax.set_title(plottitle[i], fontsize=8)
    int_month = ['Jan', 'Feb', 'Mar', 'Apr',
                 'May', 'Jun', 'Jul', 'Aug',
                 'Sep', 'Oct', 'Nov', 'Dec'][sif_integral[i]]
    ax.set_xlabel(f'Integrated SIF (Feb-{int_month}) \n[Month $\cdot$ mW/m$^2$/sr/nm]')
    
    if i == 0:
        ax.set_ylabel('Area Planted by Week 18\n[Million Acres]')
    else:
        ax.yaxis.tick_right()
    lreg = sps.linregress(xdata, ydata)
    xplot = np.array([0.9 * min(xdata), 1.1 * max(xdata)])
    ax.plot(xplot, np.array(xplot) * lreg[0] + lreg[1],
            '--', linewidth=0.75, linestyle='dashed')
            #label="r = {:0.2f}".format(lreg[2]))
    #ax.legend(ncol=2, fontsize=8)

    ax.text(0.95, 0.05, "r = {:0.2f}, p = {:0.3f}".format(lreg[2], lreg[3]),
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8)
                # xerr=sif_std[:4].sum())
ax.legend(ncol=1, fontsize=7, loc='upper left')
#fig.legend(lines, np.arange(2010, 2017), loc='center right',
#           fontsize=7, bbox_to_anchor=(1.2, 0.5))
fig.tight_layout()
plt.savefig('corn_soy_planted_sif.pdf', bbox_inches='tight')





# Crop yields and production

fig, axarr  = plt.subplots(1, 2, figsize=(4.5, 2.5))
for i, crop in enumerate(['CORN', 'SOYBEANS']):

    dfp = df_yield.query(f'Commodity == "{crop}" & Period == "YEAR"')
    
    ax = axarr[i]
    
    xdata = []
    ydata = []
    lines = []
    for j, year in enumerate(range(2010, 2017)):

        sif_data = dfs.loc[dfs.index.year == year, SL['abs_755']]
        sif_mean = sif_data.mean(axis=1)
        sif_std = sif_data.std(axis=1)

        # We are deviding by 1e8 because million * percent
        crop_data = dfp[dfp['Year'] == year]['Value'].mean()
        crop_std = dfp[dfp['Year'] == year]['Value'].std()
    
        #crop_idx = np.where(crop_data.index.week == crop_week[i])[0][0]
        
        xdata.append(sif_mean.sum())
        ydata.append(crop_data)

        ax.errorbar(x=xdata[-1],
                    y=ydata[-1],
                    xerr=np.sqrt(np.sum(sif_std**2)),
                    yerr=crop_std,
                    fmt=markerlist[j], label=year,
                    markersize=3,
                    capsize=1.0,
                    elinewidth=1.0)

    
    if i == 0:
        ax.set_ylabel('Mean crop yield \n[Bushels / acre]')
    else:
        ax.yaxis.tick_right()
        
    lreg = sps.linregress(xdata, ydata)
    xplot = np.array([0.9 * min(xdata), 1.1 * max(xdata)])
    ax.plot(xplot, np.array(xplot) * lreg[0] + lreg[1],
            '--', linewidth=0.75, linestyle='dashed')
    
    ax.text(0.95, 0.05, "r = {:0.2f}, p = {:0.3f}".format(lreg[2], lreg[3]),
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8)
        
    if i == 1:
        ax.legend(fontsize=8)
    print(sps.linregress(xdata, ydata))
