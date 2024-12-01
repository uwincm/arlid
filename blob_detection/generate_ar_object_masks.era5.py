import numpy as np
import xarray as xr
from pandas import date_range
import datetime as dt
from multiprocessing import Pool
import tqdm
import sys
import argparse

from retrieve_era5_fields_nc import retrieve_era5_fields_nc
from ar_objects_mask import *

## You need to set this for your system.
DATA_DIRECTORY='/home/orca/data/model_anal/era5/from_rda'

# Command line args:
# Must specify at least YYYYMMDDHH_start and YYYYMMDDHH_end
# Optional flags: -delta_t for time interval (hours) and -np for processors.
def getargs():
    args = {}
    parser = argparse.ArgumentParser()
    parser.add_argument("date_begin")
    parser.add_argument("date_end")
    parser.add_argument("-dt", "--delta_t", dest = "deltat", default = 1,
        help="Time interval in hours")
    parser.add_argument("-np", "--n_processors", dest = "np", default = 1,
        help="Number of processors")

    if len(sys.argv) > 1:
        args = parser.parse_args()

        # Convert from YYYYMMDDHH strings to datetime.
        fmt = '%Y%m%d%H'
        args.date_begin = dt.datetime.strptime(args.date_begin, fmt)
        args.date_end = dt.datetime.strptime(args.date_end, fmt)

        # Convert deltat and np to integers
        args.deltat = int(args.deltat)
        args.np = int(args.np)

        return args

    else:
        print(f'python {sys.argv[0]} --help for help.')
        sys.exit(1)


def process_a_time(time_to_process, verbose=False):
    """
    Process a single time. Meant to be run in parallel
    using multiprocessing.Pool.
    - Read in the ERA5 data
    - Set the basic variables
    - Calculate the masks
    - Save the output
    """

    F = retrieve_era5_fields_nc(time_to_process, DATA_DIRECTORY,
                                verbose=verbose)

    # Initialize mask object
    mask = ar_object_mask(time_to_process)

    # Set basic variables
    mask.set_coordinates(F['lon'], F['lat'], verbose=verbose)
    mask.set_topography(F['orog'])
    mask.set_tpw(F['tpw'])

    mask.set_ivt(F['viwve'], F['viwvn'])

    # Do calculations
    mask.calc_tpw_background()
    mask.calc_deep_tropics_mask()
    mask.calc_ar_mask(verbose=verbose)

    # Write the data to NetCDF
    mask.write(path_fmt='./mask_data/%Y/%m/ar%Y%m%d%H.nc',
        verbose=verbose, full_output=False)


if __name__ == '__main__':

    args = getargs()

    dt_list = date_range(start=args.date_begin,
                         end=args.date_end,
                         freq=f'{args.deltat}H')

    with Pool(args.np) as p:
        r = list(tqdm.tqdm(p.imap(process_a_time, dt_list),
                           total=len(dt_list),
                           desc = 'Creating AR objects masks'))

    print('All done!')
