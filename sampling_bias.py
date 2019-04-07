#import matplotlib
#matplotlib.use('agg')

from matplotlib import rcParams
import matplotlib.patheffects as PathEffects


rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times', 'Times New Roman']

rcParams['xtick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7
rcParams['axes.labelsize'] = 8
rcParams['axes.titlesize'] = 10
rcParams['mathtext.fontset'] = 'stix'





import h5py
from netCDF4 import Dataset
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcol
from cartopy import crs as ccrs
import cartopy.feature as cfeature

from bbox import *

# Read land cover data, as well as land cover labels
lc_nc = Dataset('ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.nc', 'r')
lc_csv = pd.read_csv('ESACCI-LC-Legend.csv', sep=';')

# Due to the long names of the LC classes, we need to manually put \n's in here
# so that they actually fit in the plot.
lc_labels = np.array([x for x in lc_csv['LCCOwnLabel'].values])
lc_labels[2] = "Rainfed cropland\n(herbaceous cover)"
lc_labels[6] = "Mosaic natural vegetation\n(tree, shrub, herbaceous cover)\n(>50%) / cropland (<50%)"
lc_labels[8] = "Tree cover, broadleaved,\ndeciduous (closed to open >15%)"
lc_labels[28] = "Tree cover, flooded,\nfresh or brakish water"
lc_labels[17] = "Tree cover, mixed leaf type\n(broadleaved and needleleaved)"


# Read the fluorescence subset
gosat_h5 = h5py.File('fluorescence_subset.h5', 'r')
# Grab the LC data
lc_gosat = gosat_h5['lccs_fraction'][:]

# Now subset the original LC data according to the bounding box
lon_min_idx = np.searchsorted(lc_nc.variables['lon'][:], lon_min)
lon_max_idx = np.searchsorted(lc_nc.variables['lon'][:], lon_max)
lat_min_idx = len(lc_nc.variables['lat']) - np.searchsorted(lc_nc.variables['lat'][:][::-1], lat_min)
lat_max_idx = len(lc_nc.variables['lat']) - np.searchsorted(lc_nc.variables['lat'][:][::-1], lat_max)

lccs_box = lc_nc.variables['lccs_class'][lat_max_idx:lat_min_idx,
                                         lon_min_idx:lon_max_idx]

lccs_lon = lc_nc.variables['lon'][lon_min_idx:lon_max_idx]
lccs_lat = lc_nc.variables['lat'][lat_max_idx:lat_min_idx]

lccs = lccs_box.flatten()

# Mask water bodies
lccs_box = np.ma.masked_equal(lccs_box, 210)

# Remove ALL water bodies, as we are not measuring over those surfaces
lccs = lccs[lccs != 210]

lc_flags = lc_nc['lccs_class'].flag_values.astype('uint8')

# Here we figure out the fractional contribution of each LC class type to the
# entire subset, both for the original bounding box, and for the GOSAT sampling
lcmax = lc_gosat[:].argmax(axis=1)
GOSAT_lcs = lc_flags[np.searchsorted(lc_flags, lcmax)]

lc_un_ALL =  np.unique(lccs, return_counts=True)

lcsum = np.mean(lc_gosat[:], axis=0) * 100

lc_percs_ALL = np.zeros_like(lcsum)
lc_percs_ALL[np.searchsorted(lc_flags, lc_un_ALL[0])] = 100 * lc_un_ALL[1] / len(lccs)
lc_sort_ALL = np.argsort(lc_percs_ALL)[::-1]

lc_un_GOSAT = np.unique(GOSAT_lcs, return_counts=True)
lc_sort_GOSAT = np.argsort(lc_un_GOSAT[1])[::-1]


# Make a nice barplot with the results
bar_width = 0.25
plt.figure(figsize=(2.5, 3), dpi=300)

plt.barh(np.arange(len(lc_percs_ALL)) - bar_width / 2,
         lc_percs_ALL[lc_sort_ALL],
         bar_width, label='Region of interest')

plt.barh(np.arange(len(lc_percs_ALL)) + bar_width / 2,
         lcsum[lc_sort_ALL],
         bar_width, label='GOSAT sampling')

plt.yticks(np.arange(len(lc_percs_ALL)),
           lc_labels[lc_sort_ALL], # rotation='vertical',
           fontsize=8)

plt.ylim(-0.5, 5.5)
plt.xlabel('Coverage [%]')
plt.legend(fontsize=8)
plt.savefig('lc_coverage.pdf', bbox_inches='tight')
plt.close()


# We want to do this on a monthly basis, so we know if we have sampling bias
# compared to a full ROI segment (bounding box)

cols_all = ['all_{:d}'.format(x) for x in range(len(lc_labels))]
cols_GOSAT = ['gosat_{:d}'.format(x) for x in range(len(lc_labels))]

df_diff = pd.DataFrame(index=pd.date_range('1/1/2010', '31/12/2016', freq='MS'),
                       columns=['max_diff', 'min_diff',] + cols_all + cols_GOSAT)

for year in range(2010, 2017):
    for month in range(1, 13):

        print(year, month)
        
        this_idx = np.where((gosat_h5['Year'][:] == year) &
                            (gosat_h5['Month'][:] == month))[0]
        
        lcmax = lc_gosat[this_idx].argmax(axis=1)
        GOSAT_lcs = lc_flags[np.searchsorted(lc_flags, lcmax)]

        lc_un_ALL =  np.unique(lccs, return_counts=True)

        lcsum = np.mean(lc_gosat[this_idx], axis=0) * 100

        lc_percs_ALL = np.zeros_like(lcsum)
        lc_percs_ALL[np.searchsorted(lc_flags, lc_un_ALL[0])] = 100 * lc_un_ALL[1] / len(lccs)
        lc_sort_ALL = np.argsort(lc_percs_ALL)[::-1]

        lc_un_GOSAT = np.unique(GOSAT_lcs, return_counts=True)
        lc_sort_GOSAT = np.argsort(lc_un_GOSAT[1])[::-1]

        this_diff = -(lc_percs_ALL[lc_sort_ALL] - lcsum[lc_sort_ALL])
        which_idx = np.argmax(np.abs(this_diff))
        max_diff = np.max(this_diff)

        df_idx = ((df_diff.index.year == year) &
                  (df_diff.index.month == month))

        for x in range(len(lc_labels)):
            df_diff.loc[df_idx, f'all_{x}'] = lc_percs_ALL[x]
            df_diff.loc[df_idx, f'gosat_{x}'] = lcsum[x]
        
        df_diff.loc[df_idx, 'min_diff'] = np.min(this_diff)
        df_diff.loc[df_idx, 'max_diff'] = np.max(this_diff)

        print(which_idx, np.min(this_diff), np.max(this_diff), np.mean(np.abs(this_diff)))

# Set this month to NaN's as we don't use it anyway
df_diff.loc['2015-01-01'] = np.nan
# Monthly deviation of LC coverage
fig = plt.figure(figsize=(3, 2), dpi=300)

for x in lc_sort_ALL[:3]:
    (df_diff[f'gosat_{x}'] - df_diff[f'all_{x}']).plot(label=lc_labels[x])
fig.legend(bbox_to_anchor=(1.65, 0.85), ncol=1, fontsize=7)
#fig.subplots_adjust(right=1.25)
fig.tight_layout()
plt.ylabel("LC coverage difference [$\%$]")
plt.xlabel("Year")
xmin, xmax = plt.gca().get_xlim()
plt.hlines([0], xmin=xmin, xmax=xmax, linestyle='dashed',
           color='grey', lw=0.5)
plt.xlim(xmin, xmax)
plt.savefig('lc_diff_monthly.pdf', bbox_inches='tight')
plt.close()
        
# Plot a map showing off the GOSAT sampling and the LC map

# Make custom colormap for LC classes, accoring to the original ESA color code
levels = np.append(lc_csv['NB_LAB'].values, 999) 
colors = [(lc_csv['R'][i]/255,
           lc_csv['G'][i]/255,
           lc_csv['B'][i]/255, 1.0) for i in range(len(lc_csv))]

cmap, norm = mcol.from_levels_and_colors(levels, colors, extend='neither')

proj = ccrs.LambertConformal(central_longitude=(lon_max+lon_min)/2.0,
                             central_latitude=(lat_max+lat_min)/2.0,
                             false_easting=0.0,
                             false_northing=0.0,
                             standard_parallels=(lat_min, lat_max))



fig = plt.figure(figsize=(6,3), dpi=600)

for p_idx in [1,2]:

    ax = plt.subplot(1, 2, p_idx, projection=proj)

    ax.set_extent([lon_min-1, lon_max+1, lat_min-1, lat_max+1])
    cl = ax.coastlines(resolution='50m')
    cl.set_rasterized(True)
    
    #lakes = ax.add_feature(cfeature.LAKES.with_scale('50m'))
    #lates.set_rasterized(True)
    
    states = ax.add_feature(cfeature.STATES.with_scale('50m'),
                            linestyle='-', linewidth=0.5)
    states.set_rasterized(True)


    if p_idx == 1:

        ax.pcolormesh(lccs_lon, lccs_lat, lccs_box[::1,::1],
                      cmap=cmap, norm=norm,
                      #extent=[lon_min, lon_max, lat_min, lat_max],
                      #origin='upper',
                      transform=ccrs.PlateCarree(),
                      rasterized=True)



        txt = ax.text(-100.0, 47.5, "ND", ha='center', va='center',
                transform=ccrs.PlateCarree())
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=1, foreground='w')])
        txt = ax.text(-101.0, 44.75, "SD", ha='center', va='center',
                transform=ccrs.PlateCarree())
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=1, foreground='w')])
        txt = ax.text(-94.50, 46.00, "MN", ha='center', va='center',
                transform=ccrs.PlateCarree())
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=1, foreground='w')])
        txt = ax.text(-99.50, 41.5, "KS", ha='center', va='center',
                transform=ccrs.PlateCarree())
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=1, foreground='w')])
        txt = ax.text(-93.50, 42.0, "IA", ha='center', va='center',
                transform=ccrs.PlateCarree())
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=1, foreground='w')])
        txt = ax.text(-98.0, 38.5, "NE", ha='center', va='center',
                transform=ccrs.PlateCarree())
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=1, foreground='w')])
        txt = ax.text(-93.0, 38.25, "MO", ha='center', va='center',
                transform=ccrs.PlateCarree())
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=1, foreground='w')])
        txt = ax.text(-90.0, 44.00, "WI", ha='center', va='center',
                transform=ccrs.PlateCarree())
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=1, foreground='w')])
        txt = ax.text(-88.75, 40.25, "IL", ha='center', va='center',
                transform=ccrs.PlateCarree())
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=1, foreground='w')])
        ax.set_title("Land cover")

    elif p_idx == 2:

        # KDE of sampling pattern

        h = ccrs.PlateCarree().transform_points(ccrs.PlateCarree(),
                                                gosat_h5['lon'][:],
                                                gosat_h5['lat'][:])[:, :2].T
        kde = sp.stats.gaussian_kde(h, bw_method=0.03)
        k = 150
        tx, ty = np.meshgrid(np.linspace(lon_min, lon_max, 2*k),
                             np.linspace(lat_min, lat_max, k))
        mesh = np.vstack((tx.ravel(), ty.ravel()))
        v = kde(mesh).reshape((k, 2*k))
        
        cmap = plt.get_cmap('rainbow')
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
        my_cmap[:, -1] = np.power(my_cmap[:, -1], 0.3)
        my_cmap = mcol.ListedColormap(my_cmap)

        ax.imshow(v,
                  cmap=my_cmap,
                  extent=[lon_min, lon_max, lat_min, lat_max],
                  vmax=np.percentile(v, 95),
                  interpolation='bicubic',
                  transform=ccrs.PlateCarree(),
                  rasterized=True)
        
        #ax.scatter(gosat_h5['lon'][:], gosat_h5['lat'][:],
        #           marker='.', s=1,
        #           transform=ccrs.PlateCarree(),
        #           #edgecolors='grey', linewidths=0.25,
        #           color='blue', rasterized=True)

        ax.set_title("GOSAT sampling")
   
    ax.gridlines(linestyle='dashed',
                 xlocs=[lon_min, lon_max],
                 ylocs=[lat_min, lat_max])

plt.savefig('coverage_map.png', bbox_inches='tight')
plt.close('all')
