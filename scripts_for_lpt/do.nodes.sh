#!/bin/bash

# Activate Python venv
source .venv/bin/activate

## Give input as YYYY
y1=$1

## Give node as second arg.
node=$2

qsub -V -pe smp 8 -q 'all.q@'$node -w e -cwd -o log.ar.$y1 -e log.ar.$y1 -N ar$y1 do.sh $y1 

## do.sh calls:
## python lpt_run.py ${y1}060100 ${y2}063023 >& log.${y1}_${y2}

exit 0
