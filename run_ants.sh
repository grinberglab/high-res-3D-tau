#!/bin/bash

if [ "$#" -ne 1 ]; then
		echo "Usage: run_ants.sh <SLICE_ID>"
		exit 0
fi

id=$1
idblk=$(printf "%04d\n" $id)

ref='/home/maryana/R_DRIVE/Experiments/1_End_of_Life/Registration/2.65x_zoom/P2865_+3_22_five_versions_Jonathan_stack_tiles/1stStep_Manual/P2865_+3_22.nii'
mov='/home/maryana/R_DRIVE/Experiments/1_End_of_Life/Registration/2.65x_zoom/P2865_+3_22_five_versions_Jonathan_stack_tiles/2ndStep_Automatic/converted_mipaved_res10_P2865_+3_22_five_versions_Jonathan_stack_tiles.nii'
#mov=~/storage2/Posdoc/AVID/AV23/AT100/full_res/AT100/heat_map/hm_map_-71/heat_map_-7.1_res10.nii
outp='/home/maryana/R_DRIVE/Experiments/1_End_of_Life/Registration/2.65x_zoom/P2865_+3_22_five_versions_Jonathan_stack_tiles/2ndStep_Automatic/ants_syn_P2865_+3_'$id'_'
out='/home/maryana/R_DRIVE/Experiments/1_End_of_Life/Registration/2.65x_zoom/P2865_+3_22_five_versions_Jonathan_stack_tiles/2ndStep_Automatic/ants_syn_P2865_+3_'$id'.nii'

antsRegistration -v -d 2 -r [$ref,$mov,1] -m meansquares[$ref,$mov,1,32] -t rigid[0.1] -c 30x30x10 -s 4x2x1vox -f 3x3x1 -l 1 -m mattes[$ref,$mov,1,32] -t Syn[0.5] -c 30x30x20 -s 3x3x1vox -f 4x2x2 -l 1 -o [$outp]
antsApplyTransforms -v -d 2 -i $mov -r $ref -n linear -t ${outp}1Warp.nii.gz -t ${outp}0GenericAffine.mat -o $out

echo $ref
echo $out

