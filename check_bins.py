# %%
import numpy as np
import netCDF4 as nc
import os
import matplotlib.pyplot as plt

YYYY = 2017
chem = 'O3'
# %%
fpath = '/home/yjj/Data/WRFGC_SSP5/imsi/'+str(YYYY)+'/'
#fpath = '/home/yjj/Data/WRFGC_SSP5/Pre/'+str(YYYY)+'/'
flist = [fname for fname in os.listdir(fpath) if fname.startswith('wrfout_d01_'+str(YYYY))]

data = np.array([])
for nf, fname in enumerate(flist):
    f = nc.Dataset(fpath + fname, 'r')
    fvar = f[chem][:].flatten()
    data = np.append(data, fvar)

# %%
path1 = '/home/yjj/Data/Airkorea/Post/O3/withnan/'+str(YYYY)+'/'
list1 = [fname for fname in os.listdir(path1) if fname.endswith('00:00.nc')]

nokrig = np.array([])
for nf, fname in enumerate(list1):
    f = nc.Dataset(path1 + fname, 'r')
    fvar = f[chem][:].flatten()
    nokrig = np.append(nokrig, fvar)
    if len(fvar)!=438:
        print(fname)

# %%
path2 = '/home/yjj/Data/Airkorea/Post/O3/krig/'+str(YYYY)+'/'
list2 = [fname for fname in os.listdir(path2) if fname.endswith('00:00.nc')]

yeskrig = np.array([])
for nf, fname in enumerate(list2):
    f = nc.Dataset(path2 + fname, 'r')
    fvar = f[chem][:].flatten()
    yeskrig = np.append(yeskrig, fvar)
    if len(fvar)!=438:
        print(fname)

# %%
plt.hist(data*1e3, bins=30, density=1, histtype='step', label='MOD           (# : ' +str(len(data))+')')
plt.hist(nokrig*1e3, bins=30, density=1, histtype='step', label='original OBS (# : ' +str(len(nokrig[~np.isnan(nokrig)]))+')')
plt.hist(yeskrig*1e3, bins=30, density=1, histtype='step', label='krigged OBS (# : ' +str(len(yeskrig))+')')

plt.xlabel('O$_3$ [ppb]', fontsize=13)
plt.ylabel('PDF', fontsize=13)
plt.title(str(YYYY), fontsize=15)
plt.xlim(0, 150)
plt.legend()

# %%
# %%
