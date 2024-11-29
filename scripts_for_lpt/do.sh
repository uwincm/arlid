#!/bin/bash

# Activate Python venv
source .venv/bin/activate

## Give input as YYYY (specify the first year)
y1=$1
y2=`expr $1 + 1`

# /usr/bin/time -v python lpt_run.lpo.arlid.py    ${y1}060100 ${y2}063023 >& log.${y1}_${y2}
/usr/bin/time -v python lpt_run.lpt.arlid.py    ${y1}060100 ${y2}063023 >& log.${y1}_${y2}
# /usr/bin/time -v python lpt_run.masks.arlid.py  ${y1}060100 ${y2}063023 >& log.${y1}_${y2}

exit 0
