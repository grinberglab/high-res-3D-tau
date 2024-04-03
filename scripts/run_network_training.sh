#!/bin/bash -l

export PYTHONPATH=$PYTHONPATH:/usr/local/bin/high-res-3D-tau

if [ "$#" -ne 2 ]; then
	echo "Usage: run_network_training.sh <CONFIG_PATH> <LEARNING_RATE>"
	exit 0  
fi

CONF_FILE=$1
LEARNING_RATE=$2

echo $CONF_FILE
echo $LEARNING_RATE

python3 /usr/local/bin/high-res-3D-tau/convnet_python3/network_trainer.py train $CONF_FILE $LEARNING_RATE
