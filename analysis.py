import h5py
import numpy as np
import pandas as pd
from netCDF4 import num2date


###############################################################################
## Change analysis parameters here
###############################################################################

resample = 'MS'
rs_apply = 'mean'
min_count = 100
min_date = '20100101'
max_date = '20161231'
anomaly_stat = 'median'

###############################################################################


######### PREPPING DATA ###########

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

abs_755_list = [x for x in df.columns if
                (('corr' in x) and ('abs' in x) and ('755' in x))]
abs_772_list = [x for x in df.columns if
                (('corr' in x) and ('abs' in x) and ('772' in x))]
rel_755_list = [x for x in df.columns if
                (('corr' in x) and ('rel' in x) and ('755' in x))]
rel_772_list = [x for x in df.columns if
                (('corr' in x) and ('rel' in x) and ('772' in x))]

# Resample the data according to resample period
dfs = df[min_date:max_date].resample(resample).apply(rs_apply)
# Save how many soundings we have
dfs['count'] = df[min_date:max_date].loc[:, 'Exposure_id'].resample(resample).count().values
# And drop all rows which are below the minimum count threshold
dfs.loc[dfs['count'] < min_count, :] = np.nan


# Create anomaly dataframe
dfa = dfs.copy()

# Anomalies for monthlies
for m, dfg in dfa.groupby(dfa.index.month):
    dfa[dfa.index.month == m] -= dfg.apply(anomaly_stat)


# Data prep finished - now make some nice plots and calculations


