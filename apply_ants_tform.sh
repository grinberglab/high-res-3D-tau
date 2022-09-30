#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Usage: apply_ants_tform.sh <SLICE_ID>"
	exit 0  
fi

id=$1
#outid=$2
echo "Slice "$id

#id=340
idblk=$(printf "%04d\n" $id)
ref='/home/maryana/R_DRIVE/Experiments/1_End_of_Life/Registration/2.65x_zoom/P2865_+3_22_five_versions_Jonathan_stack_tiles/1stStep_Manual/P2865_+3_22.nii'
mov='/home/maryana/R_DRIVE/Experiments/1_End_of_Life/Registration/2.65x_zoom/P2865_+3_22_five_versions_Jonathan_stack_tiles/2ndStep_Automatic/converted_mipaved_P2865_+3_22_five_versions_Jonathan_stack_tiles_heatmap_res10.nii'
#mov=~/storage2/Posdoc/AVID/AV23/AT100/full_res/AT100_100/heat_map/hm_map_0.1/heat_map_0.1_res10.nii
outp='/home/maryana/R_DRIVE/Experiments/1_End_of_Life/Registration/2.65x_zoom/P2865_+3_22_five_versions_Jonathan_stack_tiles/2ndStep_Automatic/ants_syn_P2865_+3_'$id'_'
out='/home/maryana/R_DRIVE/Experiments/1_End_of_Life/Registration/2.65x_zoom/P2865_+3_22_five_versions_Jonathan_stack_tiles/2ndStep_Automatic/registered_P2865_+3_22_five_versions_Jonathan_stack_tiles_heatmap_res10.nii'

antsApplyTransforms -v -d 2 -i $mov -r $ref -n NearestNeighbor -t ${outp}1Warp.nii.gz -t ${outp}0GenericAffine.mat -o $out

echo $ref
echo $out
