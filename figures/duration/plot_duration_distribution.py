import numpy as np
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

def get_lpt_system_durations(data_dir):

    duration_collect = []
    for year in range(1980,2025,5):
        year2 = year + 5
        if year == 2020:
            year2 = 2024
        fn = data_dir + f'/lpt_systems_ar_{year}060100_{year2}053123.nc'

        try:
            with xr.open_dataset(fn, decode_times=False) as ds:
                this_duration = ds['duration'].data.tolist()
                duration_collect += this_duration
        except:
            print(f' {fn}: Missing file?')

    duration_collect = np.array(duration_collect)

    return duration_collect


def get_lpt_family_durations(data_dir):

    """ First I need to read in the individual ARs for each family.
    Then get the duration based on the start (end) time of
    the first (last) AR in the family.
    """

    duration_collect = []
    for year in range(1980,2025,5):
        year2 = year + 5
        if year == 2020:
            year2 = 2024
        fn = data_dir + f'/lpt_systems_ar_{year}060100_{year2}053123.nc'

        try:
            with xr.open_dataset(fn, decode_times=False) as ds:

                lptid_individual_list = ds['lptid'].values
                lpt_family_list = sorted(np.unique(np.floor(lptid_individual_list).astype(int)))

                lptid_stitched = ds['lptid_stitched'].values
                lpt_family_stitched = np.floor(lptid_stitched).astype(int)

                timestamp_stitched = ds['timestamp_stitched'].values

                for family in lpt_family_list:

                    this_family_timestamp = timestamp_stitched[lpt_family_stitched == family]
                    this_family_timestamp_begin = this_family_timestamp[0]
                    this_family_timestamp_end = this_family_timestamp[-1]

                    # Duration in hours.
                    duration = this_family_timestamp_end - this_family_timestamp_begin
                    duration_collect.append(duration)

        except:
            print(f' {fn}: Missing file?')

    duration_collect = np.array(duration_collect)

    return duration_collect



def add_histogram_to_axis(duration_collect, ax):

    sns.histplot(duration_collect, bins=np.arange(48, 528, 24), ax=ax)
    ax.set_xlabel('Duration [h]')
    ax.set_xlim([60, 528])

    n_out_of_plot = sum(duration_collect >= 504) # Bins are left-inclusive.
    ax.fill_between([504,528],[n_out_of_plot, n_out_of_plot], color='b')
    ax.set_xticks(np.append(np.array([72,]), np.arange(120, 528, 120)))


def annotate(duration_collect, category_label, ax):

    n_tot = len(duration_collect)
    n_out_of_plot = sum(duration_collect >= 504) # Bins are left-inclusive.
    n_left = sum(duration_collect < 72.0) # Bins are right-exclusive.

    ax.text(0.99, 0.22, f'{n_out_of_plot} {category_label}\n >= 21 d.',
        ha='right', color='b', transform=ax.transAxes)
    ax.text(0.05, 0.995, f'{n_left} {category_label}\n48 - 71 h.',
        ha='left', va='top', color='b', transform=ax.transAxes)
    ax.text(0.99, 0.995, f'{n_tot} {category_label}.',
        ha='right', va='top', transform=ax.transAxes)


def add_panel_label(label_text, ax):
    ax.text(0.05, 1.02, label_text, transform=ax.transAxes)
    

################################################################################

if __name__ == '__main__':

    # Set directories
    data_dir0 = (
        '/home/orca/bkerns/projects/mjo_lpt_and_ar/'
        + 'tracking/lpt-python-public'
    )
    
    data_dir_individual_ars = (
        data_dir0 + '/ar/'
        + 'data_revision_final/ar/g0_0h/thresh1/systems'
    )

    # Get durations
    duration_list_individual_ars = get_lpt_system_durations(
        data_dir_individual_ars
    )

    duration_list_ar_families = get_lpt_family_durations(
        data_dir_individual_ars
    )

    # Make the histogram plots.
    fig = plt.figure(figsize=[4,5.5])
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    add_histogram_to_axis(duration_list_individual_ars, ax1)
    add_histogram_to_axis(duration_list_ar_families, ax2)

    ax1.set_ylim([0, 14000])
    ax2.set_ylim([0, 7500])

    annotate(duration_list_individual_ars, 'ARs', ax1)
    annotate(duration_list_ar_families, 'AR Families', ax2)

    # Labeling
    add_panel_label('a.', ax1)
    add_panel_label('b.', ax2)
    plt.tight_layout()

    # Output
    fn_out_png = 'fig7.histogram_ar_duration.png'
    fn_out_pdf = 'fig7.histogram_ar_duration.pdf'

    print(f'--> {fn_out_png}')
    fig.savefig(fn_out_png, dpi=100, bbox_inches='tight')
    print(f'--> {fn_out_pdf}')
    fig.savefig(fn_out_pdf, dpi=100, bbox_inches='tight')
