#! /usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -R yes 

module load CBI scl-rh-python
python --version
. bin/activate
python3 -m pip list
python3 convnet/network_trainer.py find_lr /wynton/home/grinberg/yiyangzhang/CNN/AT100_tutorial_for_CNN_segmentation/slidenet_2classes/configuration_avid_slidenet_2class_204px.txt
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"