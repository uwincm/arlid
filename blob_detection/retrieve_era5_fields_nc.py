import numpy as np
import xarray as xr
import datetime as dt
import cftime
import calendar
import glob
import os

def retrieve_era5_fields_nc(dt_this, verbose=True, uv200=False):
    """
    retrieve_era5_fields_nc(dt_this, verbose=True, uv200=False)
    
    Read in the data fields from ERA5 for ARLiD.
    """

    end_of_month_DD = calendar.monthrange(dt_this.year, dt_this.month)[1] # output is (day_of_week, day_of_month)
    fmt_tpw = ('/home/orca/data/model_anal/era5/from_rda/tcwv/'
        +'e5.oper.an.sfc.128_137_tcwv.ll025sc.%Y%m0100_%Y%m'+str(end_of_month_DD).zfill(2)+'23.nc')
    fn_tpw = dt_this.strftime(fmt_tpw)
    if verbose:
        print(fn_tpw)

    with xr.open_dataset(fn_tpw) as DS0:
        DS = DS0.sel(time=dt_this, method='nearest').load()
        lon = DS['longitude'].data
        lat = DS['latitude'].data
        tpw = DS['TCWV'].data

    ## IVT
    fmt_viwve = ('/home/orca/data/model_anal/era5/from_rda/viwve/'
        +'e5.oper.an.vinteg.162_071_viwve.ll025sc.%Y%m0100_%Y%m'+str(end_of_month_DD).zfill(2)+'23.nc')
    fn_viwve = dt_this.strftime(fmt_viwve)
    if verbose:
        print(fn_viwve)
    with xr.open_dataset(fn_viwve) as DS0:
        DS = DS0.sel(time=dt_this)
        viwve = DS['VIWVE'].data

    fmt_viwvn = ('/home/orca/data/model_anal/era5/from_rda/viwvn/'
        +'e5.oper.an.vinteg.162_072_viwvn.ll025sc.%Y%m0100_%Y%m'+str(end_of_month_DD).zfill(2)+'23.nc')
    fn_viwvn = dt_this.strftime(fmt_viwvn)
    if verbose:
        print(fn_viwvn)
    with xr.open_dataset(fn_viwvn) as DS0:
        DS = DS0.sel(time=dt_this)
        viwvn = DS['VIWVN'].data

    if uv200:
        fmt_v200 = ('/home/orca/data/model_anal/era5/from_rda/v_200mb/%Y/%m/'
            +'e5.oper.an.pl.128_132_v.ll025uv.%Y%m%d00_%Y%m%d23.lev200mb.6hr.nc')
        fn_v200 = dt_this.strftime(fmt_v200)
        if verbose:
            print(fn_v200)
        if os.path.exists(fn_v200):
            with xr.open_dataset(fn_v200) as DS0:
                DS = DS0.sel(time=dt_this, method='ffill')
                v200 = DS['V'].data[0,:,:]
        else:
            print(f'File not found: {fn_v200}.')
            v200 = np.nan * tpw
                
        fmt_u200 = ('/home/orca/data/model_anal/era5/from_rda/u_200mb/%Y/%m/'
            +'e5.oper.an.pl.128_131_u.ll025uv.%Y%m%d00_%Y%m%d23.lev200mb.6hr.nc')
        fn_u200 = dt_this.strftime(fmt_u200)
        if verbose:
            print(fn_u200)
        if os.path.exists(fn_u200):
            with xr.open_dataset(fn_u200) as DS0:
                DS = DS0.sel(time=dt_this, method='ffill')
                u200 = DS['U'].data[0,:,:]
        else:
            print(f'File not found: {fn_u200}.')
            u200 = np.nan * tpw

    ## Orography
    fn_orog = ('/home/orca/data/model_anal/era5/from_rda/invariant/'
        +'e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc')
    if verbose:
        print(fn_orog)
    with xr.open_dataset(fn_orog) as DS:
        orog = DS['Z'].data[0,:,:] / 9.81

    ## Create output.
    F = {}
    F['lon'] = lon
    F['lat'] = lat
    F['orog'] = orog
    if uv200:
        F['u200'] = u200
        F['v200'] = v200
        F['wspd200'] = np.sqrt(np.power(u200,2)+np.power(v200,2))

    F['tpw'] = tpw
    F['viwve'] = viwve
    F['viwvn'] = viwvn

    return F
