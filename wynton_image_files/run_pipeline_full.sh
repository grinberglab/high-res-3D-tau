#!/bin/bash -l

export PYTHONPATH=$PYTHONPATH:/home/LargeSlideScan/python/UCSFSlideScan

if [ "$#" -ne 2 ]; then
	echo "Usage: run_pipeline_full.sh <ROOT_DIR> <CONFIG_PATH>"
	exit 0  
fi

ROOT_DIR=$1
CONF_FILE=$2

echo $ROOT_DIR
echo $CONF_FILE

python /home/LargeSlideScan/python/UCSFSlideScan/pipeline/run_pipeline.py $ROOT_DIR $CONF_FILE

