### Functions to get the file name for each year of the datasets.
def get_fn_gAR_revised(year):
    data_dir = (
        '/home/orca/bkerns/projects/mjo_lpt_and_ar/tracking'
        + '/lpt-python-public/ar/data_revision_final/ar/g0_0h/thresh1/systems'
        )

    year2 = year + 5
    if year == 2020:
        year2 = 2024
    fn = (data_dir + f'/lpt_composite_mask_{year}060100_{year2}053123.nc')

    return fn




def get_fn_bk_v1(year):
    data_dir = (
        '/home/orca/bkerns/projects/mjo_lpt_and_ar/tracking'
        + '/lpt-python-public-2023/ar/data/ar/g0_0h/thresh1/systems'
        )

    year2 = year + 1
    if year == 2023:
        fn = (data_dir + f'/lpt_composite_mask_{year}060100_{year2}053123.nc')
    else:
        fn = (data_dir + f'/lpt_composite_mask_{year}060100_{year2}063023.nc')

    return fn


def get_fn_scafet(year):
    data_dir = '/home/orca/data/ar/artmip/scafet'
    fn = (data_dir + f'/MERRA2.ar_tag.SCAFET_v1.3hourly.{year}0101-{year}1231.nc')
    return fn


def get_fn_tempest_lr(year):
    data_dir = '/home/orca/data/ar/artmip/tempestLR'
    fn = (data_dir + f'/ERA5.ar_tag.TempestLR.1hr.{year}.nc')
    return fn


def get_fn_tempest_lr_m2(year):
    data_dir = '/home/orca/data/ar/artmip/tempestLR'
    fn = (data_dir + f'/MERRA2.ar_tag.TempestLR.1hr.{year}.nc')
    return fn



def get_fn_arconnect(year):
    data_dir = '/home/orca/data/ar/artmip/arconnect'
    # fn = (data_dir + f'/cmip6_MRI-EMS2-0_ssp585_r1i1p1.ar_tag.ARCONNECT_v2.6hr.{year}??????00-{year}??????00.nc')
    fn = (data_dir + f'/ERA5.ar_tag.ARCONNECT_v2.1hr.{year}????-{year}????.nc')
    return fn



# Options for each dataset.
datasets = {}

datasets['gar_revised'] = {
    'label': 'gAR_revised',
    'get_fn_func': get_fn_gAR_revised,
    'ar_mask_var': 'mask'
    }



datasets['scafet'] = {
    'label': 'scafet',
    'get_fn_func': get_fn_scafet,
    'ar_mask_var': 'ar_binary_tag'
    }

datasets['tempestlr'] = {
    'label': 'TempestLR',
    'get_fn_func': get_fn_tempest_lr,
    'ar_mask_var': 'ar_binary_tag'
    }


datasets['tempestlr_m2'] = {
    'label': 'TempestLR_merra2',
    'get_fn_func': get_fn_tempest_lr_m2,
    'ar_mask_var': 'ar_binary_tag'
    }



datasets['arconnect'] = {
    'label': 'AR-CONNECT',
    'get_fn_func': get_fn_arconnect,
    'ar_mask_var': 'ar_binary_tag'
    }

