import numpy as np 
import xarray as xr 
import datetime as dt 
import cftime
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs 


PROJ0 = ccrs.PlateCarree()
PROJ = ccrs.LambertCylindrical(central_longitude=180.0)
PLOT_AREA = [110, 250, 0, 65]


## Read in family
year1 = 2005
year2 = year1 + 1
lptid = 402.0

data_dir_families = (
    '/home/orca/bkerns/projects/mjo_lpt_and_ar/tracking/'
    + 'lpt-python-public-2024/ar.testing7.merge_split.families/'
    + 'data/ar/g0_0h/thresh1/systems'
)

data_dir_merge_split = (
    '/home/orca/bkerns/projects/mjo_lpt_and_ar/tracking/'
    + 'lpt-python-public-2024/ar.testing7.merge_split/'
    + 'data/ar/g0_0h/thresh1/systems'
)


ymdh1 = f'{year1}060100'
ymdh2 = f'{year2}063023'

fn_lpt_systems = f'{data_dir_families}/{ymdh1}_{ymdh2}/lpt_system_mask_ar.lptid{lptid:010.4f}.nc'

print(fn_lpt_systems)

with xr.open_dataset(fn_lpt_systems, use_cftime=True) as ds:
    print(ds)
    lon = ds['lon'].data
    lat = ds['lat'].data
    timestamp_family = ds['time'].data
    mask_family = ds['mask'].data


fig = plt.figure(figsize=(6.0,4.2))

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

# Add contours for entire AR family
for tt, this_timestamp in enumerate(timestamp_family):
    if tt % 6 == 0:
        for ax in ax_list:
            ax.contour(lon, lat, mask_family[tt,:,:], levels=[0.5,],
                colors = ['grey',], linewidths=0.3, transform=PROJ0)


from matplotlib.colors import ListedColormap
# color_list = np.flipud(np.loadtxt('MPL_Paired.rgb', skiprows=2))
color_list = np.loadtxt('MPL_Paired.rgb', skiprows=2)
cmap = ListedColormap(color_list)

# Individual AR system within the family.
lptid_individual_list = [402.1, 402.2, 402.3, 402.4, 402.5, 402.6]
panel_labels = ['a.','b.','c.','d.','e.','f.']
panel_label_list = []

for iiii, lptid_individual in enumerate(lptid_individual_list):

    fn_lpt_systems = f'{data_dir_merge_split}/{ymdh1}_{ymdh2}/lpt_system_mask_ar.lptid{lptid_individual:010.4f}.nc'
    print(fn_lpt_systems)

    with xr.open_dataset(fn_lpt_systems, use_cftime=True) as ds:
        lon = ds['lon'].data
        lat = ds['lat'].data
        timestamp = ds['time'].data
        mask = ds['mask'].data
        centroid_lon = ds['centroid_lon'].data
        centroid_lat = ds['centroid_lat'].data

    date_range_str = (
        timestamp[0].strftime('%m/%d')
        + ' to '
        + timestamp[-1].strftime('%m/%d')
    )

    panel_label_list += [f'{panel_labels[iiii]} AR {lptid_individual_list[iiii]} ({date_range_str})']

    # Add contours for AR
    for tt0, this_timestamp in enumerate(timestamp):
        tt = np.argwhere(timestamp_family == this_timestamp)[0]
        if tt % 6 == 0:
            ax_list[iiii].contour(lon, lat, mask[tt0,:,:], levels=[0.5,],
                colors = [cmap(1.0*tt/len(timestamp_family)),],
                linewidths=0.7, transform=PROJ0, zorder=900)
    # Add starting and ending centroids.
    ax_list[iiii].scatter(centroid_lon[0], centroid_lat[0], marker='*', s=70,
        facecolors='gold', edgecolors='k', transform=PROJ0, zorder=1000)
    ax_list[iiii].scatter(centroid_lon[-1], centroid_lat[-1], marker='x', s=50,
        c='k', linewidths=2, transform=PROJ0, zorder=1000)


def add_panel_labels(ax_list, panel_label_list):
    for ii, ax in enumerate(ax_list):
        ax.text(0.0, 1.02, panel_label_list[ii], fontsize=10, transform=ax.transAxes)

add_panel_labels(ax_list, panel_label_list)

plt.tight_layout()

cax = fig.add_axes([1.02, 0.2, 0.03, 0.6])
cbar = plt.colorbar(ScalarMappable(cmap=cmap),
             cax=cax, orientation='vertical', label='2005 - 2006')

cbar_ticks = np.arange(0.0, 1.1, 0.1)
fmt = '%m/%d'
cbar_ticklabels = [timestamp_family[int(x*(len(timestamp_family)-1))].strftime(fmt) for x in cbar_ticks]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels(cbar_ticklabels)

# plt.suptitle('AR Family 402', y=1.02)

plt.savefig('fig05.ar_family.png', dpi=200, bbox_inches='tight')
plt.savefig('fig05.ar_family.pdf', dpi=200, bbox_inches='tight')

