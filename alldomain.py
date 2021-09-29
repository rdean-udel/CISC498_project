"""
===========================================================================
predict the occurrence and magnitude of daily EOF runoff over large domains (continue)
===========================================================================
"""

# Author: Yao Hu, Chirantan Ghosh
# Date: 3/24/2020
# Update: 9/27/2021

print(__doc__)

# importing necessary libraries
import re
import numpy as np
import time

import xarray as xr
import pandas as pd
import glob
from joblib import dump
from joblib import load
import numpy as np
from copy import deepcopy 
import warnings
warnings.filterwarnings("ignore")

# physical parameters
geoVar = ['Lat', 'Lon', 'albedo12m', 'greenfrac', 'hgt_m', 'landusef', 'lu_index', 'slopecat', 'snoalb',
          'soilcbot', 'soilctop', 'soiltemp']

# accumulated variables on a daily scale
accVar = ['SWFORC', 'LWFORC', 'FSA', 'FIRA', 'IRB', 'QQSFC ', 'QQSUB', 'SHG']

# average varialbes on a dily scale
avgVar = ['TGB', 'T2MV', 'Q2MV', 'SOIL_M1', 'SOIL_M2', 'SOIL_M3', 'SOIL_M4', 'SOIL_T1', 'SOIL_T2', 'SOIL_T3', 'SOIL_T4', 'SOIL_W1', 'SOIL_W2', 
          'SOIL_W3', 'SOIL_W4', 'SOILICE', 'SOILSAT', 'ZWATTABLRT', 'SNEQV']

# calculate daily values from the accumulated ones
diffVar = ['ACCPRCP', 'ACSNOM', 'QBDRYRT']

# variables
# variables
landVar = ['SWFORC', 'LWFORC', 'ACCPRCP', 'EMISS', 'FSA', 'FIRA', 'HFX', 'LH', 'EDIR', 'ETRAN', 'ZWT', 'WA',
           'WT', 'TR', 'IRG', 'SHG', 'EVG', 'SAG', 'IRB', 'SHB', 'EVB', 'TRAD', 'TG', 'TGV', 'TGB', 'T2MV',
           'Q2MV', 'Q2MB', 'ZSNSO_SN', 'SNICE', 'SNLIQ', 'SOIL_T1', 'SOIL_W1', 'SNOW_T', 'SNOWH', 'SNEQV', 'QSNOW',
           'ISNOW', 'FSNO', 'ACSNOW', 'ACSNOM', 'CM', 'CH', 'CHV', 'CHB', 'CHLEAF', 'CHUC', 'CHV2', 'CHB2', 'RTMASS',
           'STMASS', 'WOOD', 'NEE', 'GPP', 'ACCET', 'SOILICE', 'SOILSAT', 'SNOWT_AVG', 'SOIL_T2', 'SOIL_T3', 'SOIL_T4', 'SOIL_W2', 'SOIL_W3', 'SOIL_W4']

# runoffVar = ['zwattablrt', 'QBDRYRT', 'sfcheadsubrt', 'qqsfc_acc', 'qqsub_acc', 'SOIL_M']
runoffVar = ['QBDRYRT', 'SOIL_M1', 'SOIL_M2', 'SOIL_M3', 'SOIL_M4', 'SFCHEADSUBRT', 'ZWATTABLRT']

# get the hourly values of influential variables from Model Outputs
# def getDailyValue(hourlyDataSets):
#
#     newDataSets = {}
#
#     # sum up the accumulated values
#     DateTime = pd.to_datetime(hourlyDataSets['DateTime'], format='%Y/%m/%d %H:%M')
#
#     tempdata = hourlyDataSets.drop('DateTime', 1)
#
#     for col in tempdata.columns:
#
#         subdataset = pd.DataFrame(tempdata[col].to_numpy(), index=DateTime, columns=[col])
#
#         if col in accVar:
#             subset = subdataset.groupby([lambda x: x.strftime("%y/%m/%d")]).sum()
#         elif col in accVarRate:
#             # calculate the accumulated values (60*60/day)
#             subset = subdataset.groupby([lambda x: x.strftime("%y/%m/%d")]).sum().apply(lambda x: x*60.0*60.0)
#         elif col in diffVar:
#             # calculate the difference (current day - previous day)
#             subset = subdataset.diff().groupby([lambda x: x.strftime("%y/%m/%d")]).sum()
#         else:
#             subset = subdataset.groupby([lambda x: x.strftime("%y/%m/%d")]).mean()
#
#         if len(newDataSets) == 0:
#             newDataSets = subset
#         else:
#             newDataSets = pd.concat([newDataSets, subset], axis=1)
#
#     newDataSets['DateTime'] = pd.to_datetime(subset.index, format="%y/%m/%d").strftime("%-m/%-d/%y")
#
#     newDataSets = newDataSets.reset_index(drop=True)
#
#     return newDataSets


# path to land and runoff routing files

def getDailyValue(inland, inroute, variables, infLandVar, infRunoffVar):
    
    newDataSets = {}
    # Reads in all the files and makes one long record according to 'time'
    ncland = xr.open_mfdataset(inland)
    ncroute = xr.open_mfdataset(inroute)
    # land grid i/j and routing grid x/y (1:16)
    max_i = ncland.dims['x']
    max_j = ncland.dims['y']
    max_x = ncroute.dims['x']
    max_y = ncroute.dims['y']

    if (4*max_i != max_x) or (4*max_j != max_y):
        print('The ratio of land coordinates to runoff coordinate is not 1 to 16.')
        exit()
    

    # hourly data
    landDateTime = ncland.coords['time'].values
    runoffDateTime = ncroute.coords['time'].values

    if any(landDateTime == runoffDateTime):
        print('The two timestamps are equal.')
    else:
        exit()

    for var in variables:

        if var in infLandVar:
            if var in accVar:
                subset = ncland[var].resample(time='D', skipna=True).sum()
            elif var in avgVar:
                
                s = re.findall("SOIL_W|SOIL_T", var)
                
                if s:
                    x = re.findall('\d', var)
                    a =str(s[0])
                    n = int(x[0]) - 1
                    subset = (ncland[a].sel(soil_layers_stag=n)).resample(time='D', skipna=True).mean()
                    #print(subset)
                else:
                    subset = ncland[var].resample(time='D', skipna=True).mean()
            elif var in diffVar:
                subset = ncland[var].diff('time').resample(time='D', skipna=True).sum()#mm
                #subset.to_netcdf(inp + 'ra.nc')
               # print(subset)
            else:
                # variables that take the mean of all non zero hourly values ##
                # this step replaces 0 with nans, takes the mean, turns nans back to 0
                # EMISS
                subset = ncland[var].where(ncland[var] != 0).resample(time='D', skipna=True).mean().fillna(0)
        else:
            if var in accVar:
                varName = var.lower() + '_acc'
                subset = ncroute[varName].resample(time='D', skipna=True).sum()
            elif var in avgVar:
                
                s = re.findall("SOIL_M", var)
                
                
                if s:
                    x = re.findall('\d', var)
                    a =str(s[0])
                    n = int(x[0]) - 1
                    subset = (ncroute[a].sel(soil_layers_stag=n)).resample(time='D', skipna=True).mean()
                    #print(subset)
                    
                else:
                    varName = var.lower()
                    subset = ncroute[varName].resample(time='D', skipna=True).mean()
                
            elif var in diffVar:
                subset = ncroute[var].diff('time').resample(time='D', skipna=True).sum() #mm
            else:
                varName = var.lower()  # 'sfcheadsubrt'
                subset = ncroute[varName].where(ncroute[varName] != 0).resample(time='D', skipna=True).mean().fillna(0) # mm

            # average over 16 grid cells
            #print(subset)
            subset = subset.coarsen(x=4, y=4).mean(skipna=True)
            #print(subset)
        subset = subset.rename(var)
        subset = subset[0:-1, ]

        if xr.ufuncs.isnan(subset.values).any():
            print('%s has NaN values.' % var)
            # diagnose
            # np.argwhere(np.isnan(subset.values))

        #if len(newDataSets) == 0:
         #   newDataSets = subset
        #else:
        newDataSets = xr.merge([subset, newDataSets], join="override")
        
    #print(newDataSets)
    return newDataSets

def getNormalizedValue(dailyDatasets):
    # remove 'Date Time'
    normdatasets_mean = dailyDatasets.groupby("time.day").mean("time")
    normdatasets_std = dailyDatasets.groupby("time.day").std("time")
    normdatasets= xr.apply_ufunc(
        lambda x, m, s: (x-m)/s,
        dailyDatasets.groupby("time.day"),
        normdatasets_mean,
        normdatasets_std,
        dask='allowed',
        vectorize=True
    )

    return normdatasets


def prdRunoff(file_path, datasets, variables):

    # check if all variables are in datasets1 and datasets2
    datasets_df = datasets.to_dataframe()
    colnames = datasets_df.columns.to_list()

    if not all(col in colnames for col in variables):
        print('not all variables appear in the data set')
        exit()

     
    occ_mod_name = file_path + 'clas.joblib.dat'
    cv_occ_mod = load(occ_mod_name)
    #print(cv_occ_mod)
    # data for occurrence prediction of daily EOF events
    val_c_xv = datasets_df[variables]
    #print(np.where(val_c_xv>0))
    #print(np.where(val_c_xv.values > 0))
    #print(val_c_xv.values[570][0])
    # data for magnitude prediction of daily EOF events
    val_m_xv = datasets_df[variables]
    #print(val_m_xv)
    # predict the occurrence of the EOF events using the trained boosted classification trees
    val_c_yv_mod = cv_occ_mod.predict_proba(val_c_xv)[:, 1]
    val_c_yv_ev = cv_occ_mod.predict(val_c_xv)
    runoff_event = xr.DataArray(data=val_c_yv_ev.reshape(datasets.dims['time'], datasets.dims['x'], datasets.dims['y']),
                               dims=datasets.dims, coords=datasets.coords, name='EVENT')
    #print(type(val_c_yv_mod))
    # save the results to Xarray
    runoff_prob = xr.DataArray(data=val_c_yv_mod.reshape(datasets.dims['time'], datasets.dims['x'], datasets.dims['y']),
                               dims=datasets.dims, coords=datasets.coords, name='PROB')
    #print(runoff_event)
    # load the regression model
    reg_mod_name = file_path + 'reg.joblib.dat'
    cv_reg_mod = load(reg_mod_name)

    # predict the magnitude of the EOF events using the trained boosted regression trees
    val_m_yv_mod = cv_reg_mod.predict(val_m_xv)
    val_m_yv_mod[val_m_yv_mod < 0] = 0.0
    runoff_mag = xr.DataArray(data=val_m_yv_mod.reshape(datasets.dims['time'], datasets.dims['x'], datasets.dims['y']),
                              dims=datasets.dims, coords=datasets.coords, name='RUNOFF')
    #print(runoff_mag)
    # save the results
    res_runoff = xr.merge([runoff_event, runoff_mag, runoff_prob], join="override")
    
    #print(res_runoff)
    #print(res_runoff)
    #res_runoff.rename_vars({'rainrate':'ACCPRCP', 'sfhd':'SFCHEADSUBRT'})
    return res_runoff


    

# get the hourly values of influential variables from Model Inputs

def getDailyValuesFromNWM(landOutput, runoffOutput, variables):
    # land/routing file
    # Split influential variables into the land and runoff variables
    infLandVar = []
    infRunoffVar = []

    for var in variables:
        if var in landVar:
           infLandVar.append(var)
        elif var in runoffVar:
           infRunoffVar.append(var)
        elif var.lower() in {'zwattablrt', 'sfcheadsubrt', 'qqsfc_acc', 'qqsub_acc'}:
           infRunoffVar.append(var)
        else:
            print('Influential variable: {0:s} is not included in model output'.format(var))
            exit()

    # get daily values of influential variables based on Lindsay's script
    dailyValues = getDailyValue(landOutput, runoffOutput, variables, infLandVar, infRunoffVar)

    return dailyValues


def saveData2NetCDF(inp, predictResults):
    # save Xarray to NetCDF
    predictResults.time.attrs['unit'] = 'Time Days'
    # save xarray to netCDF
    predictResults.to_netcdf(inp)


start = time.time()

print('Predict the occurrence and magnitude of daily EOF runoff:')

# Step 1: prepare the input files (Note: change the inputDir to use the script)
# read the hourly value from the netcdf file

# Provide the path where the output can be saved
#inputd = '/home/chirantan/whole_prob/'
inputd = '/C/Users/4ryan/cisc498/project/whole_prob/'

# Provide path to the input files (10 days)
#l = r"/home/chirantan/output1/*.LDASOUT_DOMAIN1"
l = r"/C/Users/4ryan/cisc498/project/nwm_outputs/*.LDASOUT_DOMAIN1"
landOutput= glob.glob(l)
#print(landOutput)
# runoff output

#r = r"/home/chirantan/output1/*.RTOUT_DOMAIN1"
r = r"/C/Users/4ryan/cisc498/project/nwm_outputs/*.RTOUT_DOMAIN1"
runoffOutput= glob.glob(r)
# influential variables

#provide path to the clusters file
#file_path = '/home/chirantan/clusters5D/'
file_path = '/C/Users/4ryan/cisc498/project/clusters5D'

v = {1: ['ACCPRCP', 'SOIL_W1', 'SFCHEADSUBRT', 'SOIL_M4', 'SOILICE'], 2: ['ACCPRCP', 'SNEQV', 'T2MV', 'SOILSAT', 'SHG', 'QQSUB_ACC'], 
             3: ['ACCPRCP', 'SHG', 'SOIL_W1', 'IRB', 'SOIL_T4', 'SOIL_M4', 'T2MV', 'SOIL_M3', 'SNEQV'], 4: ['ACCPRCP'], 5: ['ACCPRCP', 'SFCHEADSUBRT']}


for i in range(1, 6):
    x = v.get(i)
    y = deepcopy(x)
    y.reverse()
    variables=y
    

# get the hourly model output from WRF-Hydro model runs
    dailyValues = getDailyValuesFromNWM(landOutput, runoffOutput, variables)

# Step 2: load the statistical models and make predictions
# provide the path to both event and runoff prediction models (saved as classification_model.sav and regression_model.sav)
    
    f = file_path + 'cluster' + str(i) + '/'
    
    
    predictResults = prdRunoff(f, dailyValues, variables)
    
    
    inp = inputd + 'p' + str(i) + '.nc'

#print(predictResults)

# save daily values of influential variables into NetCDF format
    saveData2NetCDF(inp, predictResults)


# get the hourly model output from WRF-Hydro model runs


# Step 2: load the statistical models and make predictions
# provide the path to both event and runoff prediction models (saved as clas.joblib.dat and reg.joblib.dat)

    
v = {}
vr = {}
vp={}

#provide path to the cluster definition file and also the input path which has previously been provided to save the output
#fi = '/home/chirantan/whole_prob/'
#fil = '/home/chirantan/Cluster_1km.nc'
fi = '/C/Users/4ryan/cisc498/project/whole_prob/'
fil = '/C/Users/4ryan/cisc498/project/Cluster_1km.nc'

# print(predictResults)
for j in range (1, 6):
    f = fi + 'p' + str(j) + '.nc'
    pred =xr.open_dataset(f)
    clus =xr.open_dataset(fil)
    p_e_val = pred.EVENT.values
    p_r_val = pred.RUNOFF.values
    p_p_val = pred.PROB.values
    cval = clus.Cluster.values
    cval[cval!=j] = np.nan
    cval[cval==j] = 1
    cval_dim_new = cval.T
    b, r, c =  p_e_val.shape
    re_clus_dim_val = np.repeat(cval_dim_new[np.newaxis, :, :], b, axis=0)
#re_clus_dim_val = np.repeat(np.repeat(np.repeat(cval[np.newaxis], b, 0), 4, 1), 4, 2)
    multi_e = np.multiply(p_e_val, re_clus_dim_val)
    
    multi_r = np.multiply(p_r_val, re_clus_dim_val)
    multi_p = np.multiply(p_p_val, re_clus_dim_val)
#print(multi_e.shape)
    v[j] = multi_e
    vr[j] = multi_r
    vp[j] = multi_p
    

m = v.get(1)
mm = vr.get(1)
mmm = vp.get(1)
for h in range (2, 6):
    
    k = v.get(h)
    n = np.isnan(m)
    m[n] = k[n]
    
    kk = vr.get(h)
    nn = np.isnan(mm)
    mm[nn] = kk[nn]
    
    kkk = vp.get(h)
    nnn = np.isnan(mmm)
    mmm[nnn] = kkk[nnn]

e = deepcopy(m)
ee = deepcopy(mm)
eee = deepcopy(mmm)

# Provide path to any 1 saved output file with name p1 - p5.nc
#pd = xr.open_dataset('/home/chirantan/whole_prob/p4.nc')
pd = xr.open_dataset('/C/Users/4ryan/cisc498/project/whole_prob/p4.nc')


pd.EVENT.values = e
pd.RUNOFF.values = ee
pd.PROB.values = eee



ef = pd.transpose('time', 'y', 'x')

#Save the final file to same input path as provided previously with this new name
#ef.to_netcdf('/home/chirantan/whole_prob/pF.nc')
ef.to_netcdf('/C/Users/4ryan/cisc498/project/whole_prob/pF.nc')


#This is the code to display the result 

import netCDF4
from netCDF4 import Dataset
#import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from palettable.cartocolors.diverging import Geyser_7
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

#Input path where the previous results were saved, same input path with final saved file name in next line of input path
#indir = '/home/chirantan/whole_prob/'
indir = '/C/Users/4ryan/cisc498/project/whole_prob/'
file = indir + 'pF.nc'
fname = Dataset(file, mode='r', format="NETCDF4")
#colors = ['#998ec3', '#f1a340']  # R -> G -> B

colors = [
'#3288bd',
'#fdae61',
'#66c2a5',

'#e6f598',
'#abdda4',
'#fee08b',

'#5e4fa2',

'#d53e4f',
'#f46d43',
'#9e0142']
#n_bin = [3, 6, 10, 100]  # Discretizes the interpolation into bins


    # Create the colormap
cmap_name = 'my_list'
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=10)
variable = fname.variables['RUNOFF'][:]
c = variable

fig = plt.figure(figsize=(15, 12))
#imrt = plt.imshow(variable[0,:,:], cmap= Geyser_7.mpl_colormap)
imrt = plt.imshow(c[0,:,:], cmap= cmap)

plt.title("Prediction of Event 0", size=20)


cb = fig.colorbar(imrt, orientation='vertical', ticks = [0, 1])
cb.set_label('Event')
#plt.savefig('/home/chirantan/whole_prob/Event0.pdf', bbox_inches='tight')
plt.savefig('/C/Users/4ryan/cisc498/project/whole_prob/Event0.pdf', bbox_inches='tight')
fname.close()
end = time.time()
print("Time: {0:.4f}s".format(end - start))