import netCDF4
from netCDF4 import Dataset
#import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from palettable.cartocolors.diverging import Geyser_7
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

indir = '/home/chirantan/whole_prob/'
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
plt.savefig('/home/chirantan/whole_prob/Event0.pdf', bbox_inches='tight')
fname.close()
end = time.time()
print("Time: {0:.4f}s".format(end - start))