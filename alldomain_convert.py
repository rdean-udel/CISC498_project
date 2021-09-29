#Author : Chirantan Ghosh

import xarray as xr
from copy import deepcopy
import numpy as np



# Read the saved final output file with this name
x = xr.open_dataset('C:/Users/c/Dropbox/My PC (LAPTOP-LPQD8BR7)/Desktop/cluster_model/pF.nc')

y = x.RUNOFF.values
kk = y

#Convert the magnitude of runoff values into scale range
kk = np.where((kk < 0.1), 0, kk) 


kk = np.where((kk >= 0.1) & (kk <= 0.445567), 1, kk)

kk = np.where((kk >= 0.445718) & (kk <= 2.387600), 2, kk) 

kk = np.where((kk >= 2.390725) & (kk <= 11.577752), 3, kk) 

kk = np.where((kk >= 11.608003) & (kk <= 1712.087000), 4, kk) 


#Create a new array in reference to a confusion matrix


z = np.empty((10, 1601, 2001)) * np.nan
m = kk
e = x.PROB.values

for index, values in np.ndenumerate(e):
    a, b, c = index
    #print (a, b, c)
    if ((values > 0.75) and (m[a, b, c] == 0 or m[a, b, c] == 1)):
        z[a, b, c] = 1
    elif ((values > 0.75) and m[a, b, c]==2):
        z[a, b, c] = 2
    elif ((values > 0.75) and m[a, b, c]==3):
        z[a, b, c] = 3
    elif ((values > 0.75) and m[a, b, c]== 4):
        z[a, b, c] = 3
    elif ((values <= 0.75 and values > 0.5) and (m[a, b, c] == 0 or m[a, b, c] == 1)):
        z[a, b, c] = 1
    elif ((values <= 0.75 and values > 0.5) and m[a, b, c]==2):
        z[a, b, c]= 2
    elif ((values <= 0.75 and values > 0.5) and m[a, b, c]==3):
        z[a, b, c] = 2
    elif ((values <= 0.75 and values > 0.5) and m[a, b, c] == 4):
        z[a, b, c] = 3
    elif ((values <= 0.5 and values > 0.25) and (m[a, b, c]==0 or m[a, b, c] == 1)):
        z[a, b, c] = 1
    elif ((values <= 0.5 and values > 0.25) and m[a, b, c] ==2):
        z[a, b, c] = 1
    elif ((values <= 0.5 and values > 0.25) and m[a, b, c] == 3):
        z[a, b, c] = 2
    elif ((values <= 0.5 and values > 0.25) and m[a, b, c] == 4):
        z[a, b, c] = 2
    elif ((values <= 0.25) and (m[a, b, c]==0 or m[a, b, c] == 1)):
        z[a, b, c] = 0
    elif ((values <= 0.25) and m[a, b, c]== 2):
        z[a, b, c]= 1
    elif ((values <= 0.25) and m[a, b, c] ==3):
        z[a, b, c] = 1
    elif ((values <= 0.25) and m[a, b, c] == 4):
        z[a, b, c] = 2
    
        
za = deepcopy(z)

zmed = np.empty((1, 1601, 2001)) * np.nan
zmin = np.empty((1, 1601, 2001)) * np.nan
zmax = np.empty((1, 1601, 2001)) * np.nan

r = za
rr = r[0]
k, v, j = r.shape
for index, values in np.ndenumerate(rr):
    b, c = index
    #print(g, b, c)
    l=[]
    for m in range(k):
        l.append(r[m, b, c])
    #print(l)
    d = np.median(l)
    d1 = np.min(l)
    d2 = np.max(l)
    zmed[:, b, c]= d
    zmin[:, b, c]= d1
    zmax[:, b, c]= d2
    

# Load the variable file produced from separate code for variables   
xx = xr.open_dataset('C:/Users/c/Dropbox/My PC (LAPTOP-LPQD8BR7)/Desktop/cluster_model/day10/vF.nc')

q = xx.QSNOW.values
acm = xx.ACSNOM.values
acp = xx.ACCPRCP.values


ll = ['q', 'acm', 'acp']

for j in ll:
    globals()[f"{j+'med'}"] = np.empty((1, 1601, 2001)) * np.nan
    globals()[f"{j+'min'}"] = np.empty((1, 1601, 2001)) * np.nan
    globals()[f"{j+'max'}"] = np.empty((1, 1601, 2001)) * np.nan

    if j =='q':
        r = q
               
    elif j =='acm':
        r = acm
       
    elif j =='acp':
        r = acp
        
    rr = r[0]
    k, v, j = r.shape
    for index, values in np.ndenumerate(rr):
        b, c = index
        #print(g, b, c)
        l=[]
        for m in range(k):
            l.append(r[m, b, c])
        #print(l)
        
        
        d = np.median(l)
        d1 = np.min(l)
        d2 = np.max(l)
        
        globals()[f"{j+'med'}"][:, b, c]= d
        globals()[f"{j+'min'}"][:, b, c]= d1
        globals()[f"{j+'max'}"][:, b, c]= d2
    

c = 1601
v = 2001

# put data into a dataset
ds = xr.Dataset(
    data_vars=dict(
        risk_10days_min = (["time", "y", "x"], zmin),
        risk_10days_max = (["time", "y", "x"], zmax),
        risk_10days_median =(["time", "y", "x"], zmed),
        risk_all_10days =(["Time", "y", "x"], za),
        QSNOW_Min = (["time", "y", "x"], qmin),
        QSNOW_Max = (["time", "y", "x"], qmax),
        QSNOW_Med = (["time", "y", "x"], qmed),
        ACSNOM_Min = (["time", "y", "x"], acmmin),
        ACSNOM_Max = (["time", "y", "x"], acmmax),
        ACSNOM_Med = (["time", "y", "x"], acmmed),
        ACCPRCP_Min = (["time", "y", "x"], acpmin),
        ACCPRCP_Max = (["time", "y", "x"], acpmax),
        ACCPRCP_Med = (["time", "y", "x"], acpmed),
        
        
    ),
    coords=dict(
        y =(["y"], c),
        x =(["x"], v),
        
        
    ),
    attrs=dict(description="Risk Level and Variables value"),
)

#save the file
        
ds.to_netcdf('C:/Users/c/Dropbox/My PC (LAPTOP-LPQD8BR7)/Desktop/cluster_model/day10/days10.nc')      
        
    
        
