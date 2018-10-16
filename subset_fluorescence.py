import numpy as np
import h5py

'''
This script subsets the big fluorescence data for P and S polarisation
retrievals for the bounding box defined below, taking only sounding IDs
which occur in both files.

No other filtering (quality or otherwise) is done apart from regional
selection.
'''

h5_S = h5py.File('/Users/petersomkuti/Work/fluorescence/fluor_concatenated_S.h5')
h5_P = h5py.File('/Users/petersomkuti/Work/fluorescence/fluor_concatenated_P.h5')

intersect = np.intersect1d(h5_S['Exposure_id'][:],
                           h5_P['Exposure_id'][:])

print("{:d} common soundings.".format(len(intersect)))

srt_S = np.searchsorted(h5_S['Exposure_id'][:], intersect)
srt_P = np.searchsorted(h5_P['Exposure_id'][:], intersect)

lon_min = -102.5
lon_max = -85.5
lat_min = 37.5
lat_max = 47.5

lon = h5_S['lon'][:][srt_S]
lat = h5_S['lat'][:][srt_S]

subset_idx = np.where((lon >= lon_min) & (lon <= lon_max) &
                      (lat >= lat_min) & (lat <= lat_max))[0]

print("{:d} soundings within bounding box.".format(len(subset_idx)))

skip_list = ['a_CO2', 'a_CH4', 'a_O2', 'a_H2O', 'a_aero1', 'a_aero1_tot',
             'a_ic', 'a_ic_tot', 'a_strat', 'a_strat_tot',
             'r_akfull_755', 'r_akfull_772',
             'r_pwgts_755', 'r_pwgts_772']

subset_list_S = srt_S[subset_idx].tolist()

with h5py.File('fluorescence_subset.h5', 'w') as h5_out:

    for key in h5_S.keys():
        if key in skip_list: continue
        if 'h2o' in key: continue
        if 'shat' in key: continue
        if 'tccon_distance' in key: continue
        if 'aero' in key: continue
        if 'r_ak' in key: continue
        
        print(key)
        if (h5_S[key].shape[0] == len(h5_S['Exposure_id'])):

            if 'corr' in key:
                h5_out.create_dataset(key+'_S', data=h5_S[key][:][srt_S[subset_idx]])
                h5_out.create_dataset(key+'_P', data=h5_P[key][:][srt_P[subset_idx]])
            else:
                h5_out.create_dataset(key, data=h5_S[key][:][srt_S[subset_idx]],
                                      compression=8)
        else:
            print('..special key')
            h5_out.create_dataset(key, data=h5_S[key][:],
                                  compression=8)

        
