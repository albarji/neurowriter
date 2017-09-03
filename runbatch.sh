#!/bin/bash
#
# Runs a Jupyter notebook in batch mode
#
# Arguments:
#   $1: name of the notebook to run
#

MATPLOTLIBCONF='import matplotlib as mpl\nmpl.use("Agg")\n'

notebook=$1

generatedscript=$(mktemp -p .)

# Transform to python code
jupyter nbconvert --to script --stdout $notebook |
# Remove ipython magics
grep -v 'get_ipython().magic' |
# Nullify matplotlib
cat <(echo -e "$MATPLOTLIBCONF") - > $generatedscript

# Run script
python $generatedscript

# Delete script
rm $generatedscript

