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


## Running the Scripts

Make sure the PIP virtual environment (or an equivalent Python environment)
is activated: `source venv/bin/activate`.

The main script is `generate_ar_object_masks.era5.py`. It has several
command line arguements and options, which are managed using argparse.
Here are the arguments:
- Starting Date (required) in format YYYYMMDDHH
- Ending Date (required) in format YYYYMMDDHH
- Time interval in hours (optional, default = 1), e.g., `-dt 3` for every 3 h.
- Number of CPUs to use (optional, default = 1), e.g., `-np 8` for 8 CPUs.

There are two dependencies in this directory:
- `retrieve_era5_fields_nc.py`: Read in the ERA5 data
- `ar_objects_mask`: Python class file for calculating the masks


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
