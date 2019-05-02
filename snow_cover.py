from pyhdf.SD import SD, SDC
import numpy as np
from matplotlib import pyplot as plt
import glob
from bbox import *
from datetime import datetime as dt

flist = dict.fromkeys(range(2010, 2017))
for key in flist.keys():
    flist[key] = np.sort(glob.glob(f"/Users/petersomkuti/Downloads/MYD10CM.A{key}*.hdf"))

# Find the limits for the arrays
lat = np.linspace(90 - 0.025, -90 + 0.025, 3600)
lon = np.linspace(-180 + 0.025, 180 - 0.025, 7200)

idx_lon_min = np.argmin(np.abs(lon - lon_min))
idx_lon_max = np.argmin(np.abs(lon - lon_max))
idx_lat_min = np.argmin(np.abs(lat - lat_min))
idx_lat_max = np.argmin(np.abs(lat - lat_max))

sc_array = np.ma.zeros((12 * 7, idx_lat_min - idx_lat_max, idx_lon_max - idx_lon_min))

cnt = 0
for year in range(2010, 2017):
    for month in range(12):
        print(year, month+1)
        file = SD(flist[year][month], SDC.READ)
        sc = file.select('Snow_Cover_Monthly_CMG').get()
        sc_array[cnt, :, :] = sc[idx_lat_max:idx_lat_min, idx_lon_min:idx_lon_max]
        file.end()
        cnt += 1

sc_array[sc_array > 100] = np.ma.masked

sc_climatology = sc_array[:12].copy()
for month in range(12):
    sc_climatology[month] = np.ma.median(sc_array[month::12], axis=0)

sc_anomaly = sc_array.copy()
cnt = 0
for year in range(2010, 2017):
    for month in range(12):
        sc_anomaly[cnt] = sc_array[cnt] - sc_climatology[month]
        cnt += 1

sc_thres = 25
sc_fraction = []
times = []

cnt = 0
for year in range(2010, 2017):
    for month in range(12):
        sc_fraction.append((sc_array[cnt].data > sc_thres).sum() / (~(sc_array[cnt].mask)).sum())
        cnt += 1
        times.append(dt(year, month+1, 15))


plt.plot(times, sc_fraction); plt.show()
