import numpy as np
import h5py
from bbox import *

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
union = np.union1d(h5_S['Exposure_id'][:],
                   h5_P['Exposure_id'][:])

print("{:d} common soundings.".format(len(intersect)))
print("{:d} union soundings.".format(len(union)))

srt_S = np.searchsorted(union, h5_S['Exposure_id'][:])
srt_P = np.searchsorted(union, h5_P['Exposure_id'][:])

lon = np.zeros(len(union))
lat = np.zeros(len(union))

lon[srt_S] = h5_S['lon'][:]
lat[srt_S] = h5_S['lat'][:]
lon[srt_P] = h5_P['lon'][:]
lat[srt_P] = h5_P['lat'][:]

subset_idx = np.where((lon >= lon_min) & (lon <= lon_max) &
                      (lat >= lat_min) & (lat <= lat_max))[0]

print("{:d} soundings within bounding box.".format(len(subset_idx)))

skip_list = ['a_CO2', 'a_CH4', 'a_O2', 'a_H2O', 'a_aero1', 'a_aero1_tot',
             'a_ic', 'a_ic_tot', 'a_strat', 'a_strat_tot',
             'r_akfull_755', 'r_akfull_772',
             'r_pwgts_755', 'r_pwgts_772']

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
            
            if ('corr' in key) or (key[:2] == 'r_') or (key == 'qual_flag'):                
                this_key_S = key + '_S'
                this_key_P = key + '_P'
                this_data_S = np.zeros(len(union), dtype=h5_S[key].dtype)
                this_data_P = np.zeros(len(union), dtype=h5_S[key].dtype)
                this_data_S[:] = np.nan
                this_data_P[:] = np.nan
                

                this_data_S[srt_S] = h5_S[key][:]
                this_data_P[srt_P] = h5_P[key][:]

                h5_out.create_dataset(this_key_S, data=this_data_S[subset_idx])
                h5_out.create_dataset(this_key_P, data=this_data_P[subset_idx])
                print(f"Written {key}'s")
                
            else:
                if len(h5_S[key].shape) != 1:
                    this_data = np.zeros((len(union),) + h5_S[key].shape[1:],
                                         dtype=h5_S[key].dtype)
                else:
                    this_data = np.zeros(len(union), dtype=h5_S[key].dtype)
                try:
                    this_data[:] = np.nan
                except:
                    print(f"Could not create NaNs for {key}.")
                    
                this_data[srt_S] = h5_S[key][:]
                this_data[srt_P] = h5_P[key][:]

                h5_out.create_dataset(key, data=this_data[subset_idx])
                print(f"Written {key}")

        else:
            print('..special key')
            #h5_out.create_dataset(key, data=h5_S[key][:],
            #                      compression=8)

        
