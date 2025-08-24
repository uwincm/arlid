import numpy as np 
import xarray as xr 
import datetime as dt
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
from matplotlib.patches import Polygon
import cartopy.crs as ccrs 
import colorcet as ccet 
from scipy import ndimage
from retrieve_era5_fields_nc import *
verbose=True



# Choose a time
dt1 = dt.datetime(2003,10,20,12,0,0)

plot_area = [125, 315, -60, 60]
plot_area2 = [190, 250, 15, 55]



def get_lpt_system_file_name(year1):

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

    fn_lpt_systems = f'{data_dir}/lpt_systems_ar_{ymdh1}_{ymdh2}.nc'

    return fn_lpt_systems

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



fig = plt.figure(figsize=(7,8))
central_longitude = 180
ax1 = fig.add_subplot(3,2,1, projection=ccrs.PlateCarree(central_longitude=central_longitude)) 
ax2 = fig.add_subplot(3,2,3, projection=ccrs.PlateCarree(central_longitude=central_longitude)) 
ax3 = fig.add_subplot(3,2,5, projection=ccrs.PlateCarree(central_longitude=central_longitude)) 

ax11 = fig.add_subplot(3,2,2, projection=ccrs.PlateCarree(central_longitude=central_longitude)) 
ax22 = fig.add_subplot(3,2,4, projection=ccrs.PlateCarree(central_longitude=central_longitude)) 
ax33 = fig.add_subplot(3,2,6, projection=ccrs.PlateCarree(central_longitude=central_longitude)) 


for ax in [ax1,ax2, ax3]:
    ax.coastlines()
    ax.set_extent(plot_area, crs=ccrs.PlateCarree())

    # Add gridlines
    gridlines = ax.gridlines(draw_labels=True, linestyle='--', color='gray')

    # Custom longitude and latitude locations
    gridlines.xlocator = mticker.FixedLocator([100, 140, 180, -140, -100, -60])  # Longitudes
    gridlines.ylocator = mticker.FixedLocator(np.arange(-60,80,20))         # Latitudes

    # Customize gridline labels
    gridlines.top_labels = False  # Turn off labels at the top
    gridlines.right_labels = False  # Turn off labels on the right

    # Set label style
    gridlines.xlabel_style = {'size': 9, 'color': 'k'}
    gridlines.ylabel_style = {'size': 9, 'color': 'k'}


for ax in [ax11,ax22, ax33]:
    ax.coastlines()
    ax.set_extent(plot_area2, crs=ccrs.PlateCarree())

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


F = retrieve_era5_fields_nc(dt1, verbose=verbose)
lon, lat = np.meshgrid(F['lon'], F['lat'])
tpw = F['tpw']
viwve = F['viwve']
viwvn = F['viwvn']
ivt = np.sqrt(np.power(viwve,2)+np.power(viwvn,2))

H1 = ax1.pcolormesh(lon, lat, tpw,
    vmin=0.0, vmax=60.0, cmap='cet_rainbow4', 
    rasterized=True, transform=ccrs.PlateCarree())

H11 = ax11.pcolormesh(lon, lat, tpw,
    vmin=0.0, vmax=60.0, cmap='cet_rainbow4', rasterized=True,
    transform=ccrs.PlateCarree())

H2 = ax2.pcolormesh(lon, lat, ivt,
    vmin=0.0, vmax=1500.0, cmap='hot_r', rasterized=True,
    transform=ccrs.PlateCarree())

H22 = ax22.pcolormesh(lon, lat, ivt,
    vmin=0.0, vmax=1500.0, cmap='hot_r', rasterized=True,
    transform=ccrs.PlateCarree())


# Deep Tropics Mask
fn_ar_mask = ('/home/orca/bkerns/projects/mjo_lpt_and_ar/tracking/ar_objects/'
    + dt1.strftime('mask_data.testing/%Y/%m/ar%Y%m%d%H.nc'))
with xr.open_dataset(fn_ar_mask) as ds:
    deep_tropics_mask = ds['deep_tropics_mask'].data[0,:,:]

    ax3.contourf(lon, lat, deep_tropics_mask, levels=[0.5, 999],
        colors=['cyan',], alpha=0.5, transform=ccrs.PlateCarree())

    ax3.contour(lon, lat, deep_tropics_mask, levels=[0.5,],
        colors='b', linewidths=1.5, linestyles='-', transform=ccrs.PlateCarree())

    H3_trop = ax33.contourf(lon, lat, deep_tropics_mask, levels=[0.5, 999],
        colors=['cyan',], alpha=0.5, transform=ccrs.PlateCarree())

    ax33.contour(lon, lat, deep_tropics_mask, levels=[0.5,],
        colors='b', linewidths=1.5, linestyles='-', transform=ccrs.PlateCarree())


# AR Systems mask
year1 = 2000

fn_mask = get_lpt_composite_mask_file_name(year1)

with xr.open_dataset(fn_mask) as ds:
    ds_this_time = ds.sel(time=dt1)
    ar_system_mask = ds_this_time['mask'].data


ax3.contourf(lon, lat, ar_system_mask, levels=[0.5, 999],
    colors=['red',], alpha=0.5, transform=ccrs.PlateCarree())

ax3.contour(lon, lat, ar_system_mask, levels=[0.5,],
    colors='r', linewidths=1.0, linestyles='-', transform=ccrs.PlateCarree())


H3_ar = ax33.contourf(lon, lat, ar_system_mask, levels=[0.5, 999],
    colors=['red',], alpha=0.5, transform=ccrs.PlateCarree())

ax33.contour(lon, lat, ar_system_mask, levels=[0.5,],
    colors='r', linewidths=1.0, linestyles='-', transform=ccrs.PlateCarree())


# Color bars
xcb = 0.98
cbax1 = fig.add_axes([xcb, 0.67, 0.03, 0.2])
cb1 = plt.colorbar(H1, cax=cbax1, label='TPW [mm]')

cbax2 = fig.add_axes([xcb, 0.40, 0.03, 0.2])
cb2 = plt.colorbar(H2, cax=cbax2, label='IVT [kg m$^{-1}$ s$^{-1}$]')

cbax3_ar = fig.add_axes([xcb, 0.23, 0.03, 0.1])
cb3_ar = plt.colorbar(H3_ar, cax=cbax3_ar, label='AR')
cb3_ar.set_ticks([])

cbax3_trop = fig.add_axes([xcb, 0.13, 0.03, 0.1])
cb3_trop = plt.colorbar(H3_trop, cax=cbax3_trop, label='Tropics')
cb3_trop.set_ticks([])


def draw_box(ax, box):
    coords = [
        [box[0], box[2]],
        [box[1], box[2]],
        [box[1], box[3]],
        [box[0], box[3]],
        [box[0], box[2]],
    ]
    poly = Polygon(coords, edgecolor='k', facecolor='none', transform=ccrs.PlateCarree())
    ax.add_patch(poly)


for ax in [ax1, ax2, ax3]:
    draw_box(ax, plot_area2)

def add_panel_label(ax, label):
    ax.text(0.05, 1.05, label, transform=ax.transAxes)

add_panel_label(ax1, 'a.')
add_panel_label(ax2, 'c.')
add_panel_label(ax3, 'e.')
add_panel_label(ax11, 'b.')
add_panel_label(ax22, 'd.')
add_panel_label(ax33, 'f.')


plt.savefig('fig01.classifications.png', dpi=150, bbox_inches='tight')
plt.savefig('fig01.classifications.pdf', dpi=150, bbox_inches='tight')
