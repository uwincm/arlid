import numpy as np 
import xarray as xr 
import datetime as dt 
import cftime
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs 

## Projection stuff
PROJ0 = ccrs.PlateCarree()
PROJ = ccrs.LambertCylindrical(central_longitude=180.0)
PLOT_AREA = [115, 245, 0, 65]


# Start figure
fig = plt.figure(figsize=(6.0,4))

ax1 = fig.add_subplot(3, 2, 1, projection=PROJ)
ax2 = fig.add_subplot(3, 2, 2, projection=PROJ)
ax3 = fig.add_subplot(3, 2, 3, projection=PROJ)
ax4 = fig.add_subplot(3, 2, 4, projection=PROJ)
ax5 = fig.add_subplot(3, 2, 5, projection=PROJ)
ax6 = fig.add_subplot(3, 2, 6, projection=PROJ)

ax_list = [ax1,ax2,ax3,ax4,ax5,ax6]
for ax in ax_list:
    ax.coastlines(zorder=1000)
    ax.set_extent(PLOT_AREA, crs=PROJ0)

    # Add gridlines
    gridlines = ax.gridlines(draw_labels=True, linestyle='--', color='gray')

    # Custom longitude and latitude locations
    gridlines.xlocator = mticker.FixedLocator([120, 150, 180, -150, -120])  # Longitudes
    gridlines.ylocator = mticker.FixedLocator([10, 30, 50])         # Latitudes

    # Customize gridline labels
    gridlines.top_labels = False  # Turn off labels at the top
    gridlines.right_labels = False  # Turn off labels on the right

    # Set label style
    gridlines.xlabel_style = {'size': 9, 'color': 'k'}
    gridlines.ylabel_style = {'size': 9, 'color': 'k'}

##
## Split Case
##

dt_before_split = dt.datetime(2005, 12, 18, 6, 0, 0)
dt_split_time = dt.datetime(2005, 12, 18, 13, 0, 0)
dt_after_split = dt.datetime(2005, 12, 18, 18, 0, 0)

lptid1 = 402.1  # Went to the right, as the larger peice. 
lptid2 = 402.2

## Read in data
year1 = 2005
year2 = year1 + 1

data_dir_merge_split = (
    '/home/orca/bkerns/projects/mjo_lpt_and_ar/tracking/'
    + 'lpt-python-public-2024/ar.testing7.merge_split/'
    + 'data/ar/g0_0h/thresh1/systems'
)

col1 = '#fc8d59' 
col2 = '#91bfdb'

ymdh1 = f'{year1}060100'
ymdh2 = f'{year2}063023'

fn_lpt_systems1 = f'{data_dir_merge_split}/{ymdh1}_{ymdh2}/lpt_system_mask_ar.lptid{lptid1:010.4f}.nc'
fn_lpt_systems2 = f'{data_dir_merge_split}/{ymdh1}_{ymdh2}/lpt_system_mask_ar.lptid{lptid2:010.4f}.nc'


# Pre-split
with xr.open_dataset(fn_lpt_systems1, use_cftime=True) as ds:
    ds2 = ds.sel(time=dt_before_split)
    lon = ds2['lon'].data
    lat = ds2['lat'].data
    clon = ds2['centroid_lon'].data
    clat = ds2['centroid_lat'].data
    mask = ds2['mask'].data

ax1.contourf(lon, lat, mask, levels=[0.5, 1.5], colors=col1,
    transform=PROJ0)
ax1.scatter(clon, clat, marker='^', s=50, edgecolor='k', facecolor='none',
    transform=PROJ0)

# Split Time
with xr.open_dataset(fn_lpt_systems1, use_cftime=True) as ds:
    ds2 = ds.sel(time=dt_split_time)
    lon = ds2['lon'].data
    lat = ds2['lat'].data
    clon = ds2['centroid_lon'].data
    clat = ds2['centroid_lat'].data

    mask = ds2['mask'].data

ax3.contourf(lon, lat, mask, levels=[0.5, 1.5], colors=col1,
    transform=PROJ0)
ax3.scatter(clon, clat, marker='^', s=50, edgecolor='k', facecolor='none',
    transform=PROJ0)


with xr.open_dataset(fn_lpt_systems2, use_cftime=True) as ds:
    ds2 = ds.sel(time=dt_split_time)
    lon = ds2['lon'].data
    lat = ds2['lat'].data
    clon = ds2['centroid_lon'].data
    clat = ds2['centroid_lat'].data

    mask = ds2['mask'].data

ax3.contourf(lon, lat, mask, levels=[0.5, 1.5], colors=col2,
    transform=PROJ0)
ax3.scatter(clon, clat, marker='o', s=50, edgecolor='k', facecolor='none',
    transform=PROJ0)


# After split Time
with xr.open_dataset(fn_lpt_systems1, use_cftime=True) as ds:
    ds2 = ds.sel(time=dt_after_split)
    lon = ds2['lon'].data
    lat = ds2['lat'].data
    clon = ds2['centroid_lon'].data
    clat = ds2['centroid_lat'].data

    mask = ds2['mask'].data

ax5.contourf(lon, lat, mask, levels=[0.5, 1.5], colors=col1,
    transform=PROJ0)
ax5.scatter(clon, clat, marker='^', s=50, edgecolor='k', facecolor='none',
    transform=PROJ0)


with xr.open_dataset(fn_lpt_systems2, use_cftime=True) as ds:
    ds2 = ds.sel(time=dt_after_split)
    lon = ds2['lon'].data
    lat = ds2['lat'].data
    clon = ds2['centroid_lon'].data
    clat = ds2['centroid_lat'].data

    mask = ds2['mask'].data

ax5.contourf(lon, lat, mask, levels=[0.5, 1.5], colors=col2,
    transform=PROJ0)
ax5.scatter(clon, clat, marker='o', s=50, edgecolor='k', facecolor='none',
    transform=PROJ0)




def add_panel_labels(ax_list, panel_label_list):
    for ii, ax in enumerate(ax_list):
        ax.text(0.05, 1.02, panel_label_list[ii], transform=ax.transAxes)

ax_list = [ax1,ax3,ax5]
fmt = '%Y-%m-%d %H00Z'
dt_str1 = dt_before_split.strftime(fmt)
dt_str2 = dt_split_time.strftime(fmt)
dt_str3 = dt_after_split.strftime(fmt)
panel_label_list = [f'a. {dt_str1}',f'b. {dt_str2}',f'c. {dt_str3}']
add_panel_labels(ax_list, panel_label_list)

##
## Merger Case
##

dt_before_merge = dt.datetime(2003, 11, 19, 0, 0, 0)
dt_merge_time = dt.datetime(2003, 11, 19, 7, 0, 0)
dt_after_merge = dt.datetime(2003, 11, 19, 12, 0, 0)

lptid1 = 333.1  # Absorbed 
lptid2 = 333.2  # Went to the N into Alaska

## Read in data
year1 = 2003
year2 = year1 + 1

data_dir_merge_split = (
    '/home/orca/bkerns/projects/mjo_lpt_and_ar/tracking/'
    + 'lpt-python-public-2024/ar.testing7.merge_split/'
    + 'data/ar/g0_0h/thresh1/systems'
)

ymdh1 = f'{year1}060100'
ymdh2 = f'{year2}063023'

fn_lpt_systems1 = f'{data_dir_merge_split}/{ymdh1}_{ymdh2}/lpt_system_mask_ar.lptid{lptid1:010.4f}.nc'
fn_lpt_systems2 = f'{data_dir_merge_split}/{ymdh1}_{ymdh2}/lpt_system_mask_ar.lptid{lptid2:010.4f}.nc'


# Pre-merge
with xr.open_dataset(fn_lpt_systems1, use_cftime=True) as ds:
    ds2 = ds.sel(time=dt_before_merge)
    lon = ds2['lon'].data
    lat = ds2['lat'].data
    clon = ds2['centroid_lon'].data
    clat = ds2['centroid_lat'].data
    mask = ds2['mask'].data

ax2.contourf(lon, lat, mask, levels=[0.5, 1.5], colors=col1,
    transform=PROJ0)
ax2.scatter(clon, clat, marker='^', s=50, edgecolor='k', facecolor='none',
    transform=PROJ0)

with xr.open_dataset(fn_lpt_systems2, use_cftime=True) as ds:
    ds2 = ds.sel(time=dt_before_merge)
    lon = ds2['lon'].data
    lat = ds2['lat'].data
    clon = ds2['centroid_lon'].data
    clat = ds2['centroid_lat'].data

    mask = ds2['mask'].data

ax2.contourf(lon, lat, mask, levels=[0.5, 1.5], colors=col2,
    transform=PROJ0)
ax2.scatter(clon, clat, marker='o', s=50, edgecolor='k', facecolor='none',
    transform=PROJ0)


# Merge Time
with xr.open_dataset(fn_lpt_systems1, use_cftime=True) as ds:
    ds2 = ds.sel(time=dt_merge_time)
    lon = ds2['lon'].data
    lat = ds2['lat'].data
    clon = ds2['centroid_lon'].data
    clat = ds2['centroid_lat'].data

    mask = ds2['mask'].data

ax4.contourf(lon, lat, mask, levels=[0.5, 1.5], colors=col1,
    transform=PROJ0)
ax4.scatter(clon, clat, marker='^', s=50, edgecolor='k', facecolor='none',
    transform=PROJ0)


with xr.open_dataset(fn_lpt_systems2, use_cftime=True) as ds:
    ds2 = ds.sel(time=dt_merge_time)
    lon = ds2['lon'].data
    lat = ds2['lat'].data
    clon = ds2['centroid_lon'].data
    clat = ds2['centroid_lat'].data

    mask = ds2['mask'].data

ax4.contourf(lon, lat, mask, levels=[0.5, 1.5], colors=col2,
    transform=PROJ0)
ax4.scatter(clon, clat, marker='o', s=50, edgecolor='k', facecolor='none',
    transform=PROJ0)


# After merge Time

with xr.open_dataset(fn_lpt_systems2, use_cftime=True) as ds:
    ds2 = ds.sel(time=dt_after_merge)
    lon = ds2['lon'].data
    lat = ds2['lat'].data
    clon = ds2['centroid_lon'].data
    clat = ds2['centroid_lat'].data

    mask = ds2['mask'].data

ax6.contourf(lon, lat, mask, levels=[0.5, 1.5], colors=col2,
    transform=PROJ0)
ax6.scatter(clon, clat, marker='o', s=50, edgecolor='k', facecolor='none',
    transform=PROJ0)

def add_panel_labels(ax_list, panel_label_list):
    for ii, ax in enumerate(ax_list):
        ax.text(0.05, 1.02, panel_label_list[ii], transform=ax.transAxes)

ax_list = [ax2,ax4,ax6]
fmt = '%Y-%m-%d %H00Z'
dt_str1 = dt_before_merge.strftime(fmt)
dt_str2 = dt_merge_time.strftime(fmt)
dt_str3 = dt_after_merge.strftime(fmt)
panel_label_list = [f'd. {dt_str1}',f'e. {dt_str2}',f'f. {dt_str3}']
add_panel_labels(ax_list, panel_label_list)



## Formatting
plt.tight_layout(w_pad=-0.5, h_pad=1.0)


## Output
plt.savefig('fig04.show_merge_split.png', dpi=100, bbox_inches='tight')
plt.savefig('fig04.show_merge_split.pdf', dpi=100, bbox_inches='tight')
