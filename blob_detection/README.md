# Atmospheric River Lifecycle Detectin (ARLiD): Blob Detection Step

Brandon Kerns (bkerns@uw.edu)

This directory contains the code used to generate IVT blobs, TPW blobs,
AR objects, and deep tropics mask from ERA5 data for ARLiD.


## Setting up the Python environment

A `requirements.txt` file is included to help set up a Python
virtual environment using PIP. You should be able to generate the 
virtual environment like this:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

I used Python 3.11 for this. The Python 3.11 came Anaconda. When I tried running
the code with my system Python (3.7.4), I got errors. Here are the steps I
performed to get it to work using Python 3.11 from Anaconda:
```
conda activate meteo_3.11  # Activate an environment that uses Python 3.11
python -m venv venv        # Create venv using Anaconda's Python 3.11
conda deactivate           # I only wanted the version of Python, so deactivate
source venv/bin/activate
pip install -r requirements.txt
```


## Running the Scripts

Make sure the PIP virtual environment (or an equivalent Python environment)
is activated: `source venv/bin/activate`.

There are two dependencies in this directory:
- `retrieve_era5_fields_nc.py`: Read in the ERA5 data
- `ar_objects_mask`: Python class file for calculating the masks

The main script is `generate_ar_object_masks.era5.py`. It has several
command line arguements and options, which are managed using argparse.
Here are the arguments:
- Starting Date (required) in format YYYYMMDDHH
- Ending Date (required) in format YYYYMMDDHH
- Time interval in hours (optional, default = 1), e.g., `-dt 3` for every 3 h.
- Number of CPUs to use (optional, default = 1), e.g., `-np 8` for 8 CPUs.

As an __example__, to run the script for 0000 UTC 1 January 2020, run it like this:
```
python generate_ar_object_masks.era5.py 2020010100 2020010100
```
Check that the file `mask_data/2020/01/ar2020010100.nc` was created and 
is readable. To run it for the entire day, with 3 hourly intervals, on 4 CPUs:
```
python generate_ar_object_masks.era5.py 2020010100 2020010123 -dt 3 -np 4
```
This should create 8 files:
```
$ ls -lh  mask_data/2020/01/*.nc
-rw-r--r-- 1 bkerns atgstaff 75K Dec  1 13:56 mask_data/2020/01/ar2020010100.nc
-rw-r--r-- 1 bkerns atgstaff 74K Dec  1 13:56 mask_data/2020/01/ar2020010103.nc
-rw-r--r-- 1 bkerns atgstaff 74K Dec  1 13:56 mask_data/2020/01/ar2020010106.nc
-rw-r--r-- 1 bkerns atgstaff 74K Dec  1 13:56 mask_data/2020/01/ar2020010109.nc
-rw-r--r-- 1 bkerns atgstaff 74K Dec  1 13:56 mask_data/2020/01/ar2020010112.nc
-rw-r--r-- 1 bkerns atgstaff 74K Dec  1 13:56 mask_data/2020/01/ar2020010115.nc
-rw-r--r-- 1 bkerns atgstaff 75K Dec  1 13:56 mask_data/2020/01/ar2020010118.nc
-rw-r--r-- 1 bkerns atgstaff 75K Dec  1 13:56 mask_data/2020/01/ar2020010121.nc
```
If this is successful, you should be able to run it on any time range.


## Input and Output

### Input 
ERA-5 data were obtained using the NSF NCAR/UCAR Research Data Archive
([RDA](https://rda.ucar.edu/)), dataset number 633.0.
The files needed start with `e5.oper.an.sfc.128_137_tcwv.ll025sc`
for TPW (total column water vapor, tcwv), and for IVT you need
the two components with files starting like `e5.oper.an.vinteg.162_071_viwve`
 and `e5.oper.an.vinteg.162_072_viwvn`. I used the download scripts generated
by the RDA web application to download the data.
The input directory is to be set in the script
`generate_ar_object_masks.era5.py`.


### Output 
Running this code will generate the outputs under the `mask_data/` directory.
Subdirectories will be created by year, month, and day.


## Post Processing for Zenodo

The data were tar, gzipped for each calendar year before uploading to zenodo.
The command used for this was like: `tar -cvf ar_blob_data_1980.tar 1980`.
