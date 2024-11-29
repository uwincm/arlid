# Atmospheric River Lifecycle Detectin (ARLiD): Tracking Step

Brandon Kerns (bkerns@uw.edu)

This directory contains scripts used to run the
Large Scale Precipitation Tracking (LPT) code for AR tracking.

__Suggested way to run it__: If you cloned the repository with the `--recursive` option,
you should have a copy of the LPT code under `lpt-python-public`. In that directory, do the following:
```
cp -r MASTER_RUN ar
cd ar
cp ../scripts_for_lpt/*.py .
cp ../scripts_for_lpt/*.sh .
```

LPT can be run by either using the "lpt_run" scripts and specifying the
date range on the command line, or by using the "do" scripts.


## 1. Python driver scripts "lpt_run"

- `lpt_run.lpo.arlid.py`: For LPO step. Saves data to: data/ar/g0_0h/thresh1/objects
- `lpt_run.lpt.arlid.py`: Do the LPT step. Saves data to: data/ar/g0_0h/thresh1/systems
- `lpt_run.masks.arlid.py`: Generate the spatio-temporal mask files.
  Saves data to Saves data to: data/ar/g0_0h/thresh1/systems.

Notes:
- The scripts assume the AR blob mask data are under `../blob_detection/mask_data`.
  If needed, edit the lines like this:
  ```
  dataset['raw_data_parent_dir'] = '../blob_detection/mask_data'
  ```

- The scripts are set to run with 8 CPUs. If you want to change this, edit these lines:
  ```
  lpo_options['lpo_calc_n_cores'] = 8         # Number of cores to use for LPO step.
  lpo_options['lpo_calc_n_cores'] = 8         # Number of cores to use for LPO step.
  lpt_options['mask_n_cores'] = 8             # How many processors to use for LPT system mask calculations.

  ```

## 2. Shell scripts "do"

These are the scripts I used for managing the workflow for the multiple years.

- `do.sh`: Runs the "lpt_run" script for a single tracking year (1 June to 30 June the following year), in the current terminal.
- `do.nodes.sh`: Submits an instance of `do.sh` to a compute node using the `qsub` command. This is witten for the SGE job scheduler.
- `do.submit.sh`: A script to send jobs for all the years to various compute nodes.

Notes:
- You may need to edit the number of nodes and/or adjust the syntax for your job scheduler in `do.nodes.sh`.