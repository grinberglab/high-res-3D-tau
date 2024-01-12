#!/bin/bash -l

export PYTHONPATH=$PYTHONPATH:/home/LargeSlideScan/python/UCSFSlideScan

if [ "$#" -ne 2 ]; then
        echo "Usage: run_extract_patches.sh <ROOT_DIR> <PATCHES_DIR>"
        exit 0
fi

ROOT_DIR=$1
#PATCHES_DIR="${ROOT_DIR}"'/patches_orig'
PATCHES_DIR=$2

echo $ROOT_DIR
echo $PATCHES_DIR

mkdir "${PATCHES_DIR}"
python /home/LargeSlideScan/python/UCSFSlideScan/dataset_creator/random_sampling_simple2.py $ROOT_DIR 1024 1024 $PATCHES_DIR

