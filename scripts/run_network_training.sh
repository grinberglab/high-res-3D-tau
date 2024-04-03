#!/bin/bash -l

export PYTHONPATH=$PYTHONPATH:/usr/local/bin/high-res-3D-tau

if [ "$#" -ne 1 ]; then
	echo "Usage: run_network_training.sh <CONFIG_PATH>"
	exit 0  
fi

CONF_FILE=$1

echo $CONF_FILE

python3 /usr/local/bin/high-res-3D-tau/convnet_python3/network_segmentation.py train $CONF_FILE
