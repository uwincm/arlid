import numpy as np 
import xarray as xr 
import datetime as dt
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
from matplotlib.patches import Polygon
import cartopy.crs as ccrs 
import colorcet as ccet 
from scipy import ndimage
from scipy.ndimage import convolve, minimum_filter

from retrieve_era5_fields_nc import *

# Choose a time
dt1 = dt.datetime(2003,10,20,12,0,0)

# plot_area = [155, 245, 5, 55]
# plot_area = [125, 315, -60, 60]
plot_area = [190, 250, 15, 55]

verbose = True


def get_lpt_composite_mask_file_name(year1):

    if year1 == 2020:
        year2 = year1 + 4
    else:
        year2 = year1 + 5

    data_dir = (
        '/home/orca3/bkerns/projects/mjo_lpt_and_ar/tracking/'
        + 'lpt-python-public/ar/data_revision_final/ar/g0_0h/thresh1/systems'
    )

    ymdh1 = f'{year1}060100'
    ymdh2 = f'{year2}053123'

    fn_composite_mask = f'{data_dir}/lpt_composite_mask_{ymdh1}_{ymdh2}.nc'

    return fn_composite_mask



F = retrieve_era5_fields_nc(dt1, verbose=verbose)
lon, lat = np.meshgrid(F['lon'], F['lat'])
tpw = F['tpw']
viwve = F['viwve']
viwvn = F['viwvn']
topography = F['orog']

ivt = np.sqrt(np.power(viwve,2)+np.power(viwvn,2))
area = np.abs(111.0*111.0*0.25*0.25*np.cos(np.pi*lat/180.0))

ivt_filtered = ndimage.gaussian_laplace(ivt, [5,5], truncate=2)

# Deep Tropics Mask and TPW background
fn_ar_mask = ('/home/orca/bkerns/projects/mjo_lpt_and_ar/tracking/ar_objects/'
    + dt1.strftime('mask_data.testing7/%Y/%m/ar%Y%m%d%H.nc'))
print(fn_ar_mask)
with xr.open_dataset(fn_ar_mask) as ds:
    deep_tropics_mask = ds['deep_tropics_mask'].data[0,:,:]



mask_ivt = ivt_filtered < -7.0
mask_ivt = ndimage.binary_fill_holes(mask_ivt)


def calc_tpw_background(tpw, topography):
    
    rad = int(5.0/.25)
    MAXTOPO = 500.0
    # MAXTOPO = 1000.0

    if tpw.ndim == 2:

        n_pad_y_lat = 1*rad
        tpw_padded_lat = np.append(tpw[0:n_pad_y_lat,:], tpw, axis=0)
        tpw_padded_lat = np.append(tpw_padded_lat, tpw[-1*n_pad_y_lat:,:], axis=0)
        topo_padded_lat = np.append(topography[0:n_pad_y_lat,:], topography, axis=0)
        topo_padded_lat = np.append(topo_padded_lat, topography[-1*n_pad_y_lat:,:], axis=0)

        #Effectively disregard points above the MAXTOPO elevation.
        tpw_topo_max = np.where(topo_padded_lat > MAXTOPO, 999.0, tpw_padded_lat)

        # Set the shapes used for the minimum filters
        shape_left = np.append(np.ones([1,rad]), np.zeros([1,rad+1]), axis=1)
        shape_right = np.append(np.zeros([1,rad+1]), np.ones([1,rad]), axis=1)
        shape_top = np.append(np.ones([rad,1]), np.zeros([rad+1,1]), axis=0)
        shape_bottom = np.append(np.zeros([rad+1,1]), np.ones([rad,1]), axis=0)

        # Calculate the minimum filters.
        # I'm using "wrap" mode here, but I actually only want it to wrap in the 
        # zonal direction. So I pad it in the meridional direction
        # Then remove the artificial points at the north and south edges.
        tpw_filter_left = minimum_filter(tpw_topo_max, footprint=shape_left, mode='wrap')
        tpw_filter_right = minimum_filter(tpw_topo_max, footprint=shape_right, mode='wrap')
        tpw_filter_top = minimum_filter(tpw_topo_max, footprint=shape_top, mode='wrap')
        tpw_filter_bottom = minimum_filter(tpw_topo_max, footprint=shape_bottom, mode='wrap')

        # Combine the minimum filters to get the full background TPW.
        tpw_filter_left_right = np.maximum(tpw_filter_left, tpw_filter_right)
        tpw_filter_top_bottom = np.maximum(tpw_filter_top, tpw_filter_bottom)

        tpw_filter_all = np.minimum(tpw_filter_left_right, tpw_filter_top_bottom)

        # Throw away the data where I stitched at the north and south edges.
        tpw_filter_all = tpw_filter_all[n_pad_y_lat:-1*n_pad_y_lat,:]

        # I don't have a background TPW for high topography.
        # Therefore, set it to the original TPW.
        # That way when I calculate the difference from the background,
        # I will get zero.
        tpw_filter_all = np.where(tpw_filter_all > 998.0, tpw, tpw_filter_all)

        # Set the background TPW.
        return tpw_filter_all

tpw_background = calc_tpw_background(tpw, topography)


## Plotting
fig = plt.figure(figsize=[7,4.5])

central_longitude = 180
ax1 = fig.add_subplot(2,2,1, projection=ccrs.PlateCarree(central_longitude=central_longitude)) 
ax2 = fig.add_subplot(2,2,2, projection=ccrs.PlateCarree(central_longitude=central_longitude)) 
ax3 = fig.add_subplot(2,2,3, projection=ccrs.PlateCarree(central_longitude=central_longitude)) 
ax4 = fig.add_subplot(2,2,4, projection=ccrs.PlateCarree(central_longitude=central_longitude)) 

for ax in [ax1,ax2,ax3, ax4]:
    ax.coastlines()
    ax.set_extent(plot_area, crs=ccrs.PlateCarree())

    # Add gridlines
    gridlines = ax.gridlines(draw_labels=True, linestyle='--', color='gray')

    # Custom longitude and latitude locations
    gridlines.xlocator = mticker.FixedLocator(np.arange(-180, -80, 20))  # Longitudes
    gridlines.ylocator = mticker.FixedLocator(np.arange(10,70,10))         # Latitudes

    # Customize gridline labels
    gridlines.top_labels = False  # Turn off labels at the top
    gridlines.right_labels = False  # Turn off labels on the right

    # Set label style
    gridlines.xlabel_style = {'size': 9, 'color': 'k'}
    gridlines.ylabel_style = {'size': 9, 'color': 'k'}

H1 = ax1.pcolormesh(lon, lat, ivt_filtered, vmin=-15, vmax=15, cmap='cet_bwy',
    rasterized=True, transform=ccrs.PlateCarree())

ax1.contour(lon, lat, ivt_filtered, levels=[-7.0,], colors='k',
    linewidths=0.7, linestyles='-', transform=ccrs.PlateCarree())

cax=fig.add_axes([0.20, 0.95, 0.25, 0.03])
plt.colorbar(H1, cax=cax, orientation='horizontal', location='top',
    label='LoG Filtered IVT [kg m$^{-1}$ s$^{-1}$ deg$^{-2}$]')

# TPW

H2 = ax2.pcolormesh(lon, lat, tpw - tpw_background,
    vmin=-15.0, vmax=15.0, cmap='cet_gwv', rasterized=True,
    transform=ccrs.PlateCarree())

tpw_mask1 = tpw - tpw_background > 10.0
tpw_mask2 = tpw - tpw_background > 0.5*tpw_background
tpw_mask = np.logical_or(tpw_mask1, tpw_mask2)

ax2.contour(lon, lat, tpw_mask,
    levels=[0.5,], colors='k', linewidths=0.7, transform=ccrs.PlateCarree())

cax=fig.add_axes([0.65, 0.95, 0.25, 0.03])
plt.colorbar(H2, cax=cax, orientation='horizontal', location='top',
    label='TPW Diff. [mm]')

ax2.contourf(lon, lat, topography, levels=[1000, 9999], colors=['none',], 
    hatches=['////',], transform=ccrs.PlateCarree())


## Mask showing the final IVT mask

# IVT Mask
H3 = ax3.contourf(lon, lat, mask_ivt, levels=[-999.0,0.5,999.0],
    colors = ['none','darkgrey'], transform=ccrs.PlateCarree(), zorder=100)

## TPW Mask

# IVT Mask
H4 = ax4.contourf(lon, lat, tpw_mask, levels=[-999.0,0.5,999.0],
    colors = ['none','darkgrey'], transform=ccrs.PlateCarree(), zorder=100)

ax4.contourf(lon, lat, topography, levels=[1000, 9999],
    colors=['none',], hatches=['////',], transform=ccrs.PlateCarree())


year1 = 2000 #dt1.year
fn_mask = get_lpt_composite_mask_file_name(year1)
with xr.open_dataset(fn_mask) as ds:
    ds_this_time = ds.sel(time=dt1)
    ar_system_mask = ds_this_time['mask'].data

for ax in [ax1,ax2,ax3,ax4]:
    ax.contour(lon, lat, deep_tropics_mask, levels=[0.5,],
        colors=['k',], linestyles='--', transform=ccrs.PlateCarree())

for ax in [ax3,ax4]:
    ax.contourf(lon, lat, ar_system_mask, levels=[0.5, 999],
        colors=['red',], alpha=0.5, transform=ccrs.PlateCarree(), zorder=200)

    ax.contour(lon, lat, ar_system_mask, levels=[0.5,],
        colors=['r',], linewidths=1, transform=ccrs.PlateCarree())

def add_panel_label(ax, label):
    ax.text(0.05, 1.05, label, transform=ax.transAxes)

add_panel_label(ax1, 'a.')
add_panel_label(ax2, 'b.')
add_panel_label(ax3, 'c.')
add_panel_label(ax4, 'd.')

plt.tight_layout(h_pad=0.5, w_pad=-3.0)

plt.savefig('fig03.ivt_and_tpw_mask.png', dpi=150, bbox_inches='tight')
plt.savefig('fig03.ivt_and_tpw_mask.pdf', dpi=150, bbox_inches='tight')
