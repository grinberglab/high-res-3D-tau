#!/bin/bash -l

export PYTHONPATH=$PYTHONPATH:/usr/local/bin/high-res-3D-tau

if [ "$#" -ne 2 ]; then
	echo "Usage: run_pipeline_full.sh <ROOT_DIR> <CONFIG_PATH>"
	exit 0  
fi

ROOT_DIR=$1
CONF_FILE=$2

echo $ROOT_DIR
echo $CONF_FILE

python3 /usr/local/bin/high-res-3D-tau/pipeline/run_pipeline_tilemask.py $ROOT_DIR $CONF_FILE
