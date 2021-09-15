#!/bin/bash

if [ "$#" -ne 5 ]; then
	echo "Usage: apply_ants_tform.sh <MOV_IMG> <REF_IMG> <SLICE_NUM> <REG_FOLDER> <OUT_IMG>"
	exit 0  
fi

mov=$1
ref=$2
id=$3
reg_dir=$4
out_img=$5
echo "Slice "$id

#id=340
#idblk=$(printf "%04d\n" $id)
#ref=../2ndStep_Automatic/'1181_001-Whole-Brain_'$idblk'.png.nii'
#mov=../2ndStep_Automatic/'converted_transformed_AV1AT8_'$id'_norm_heatmap_'$outid'.nii'
#mov=~/storage2/Posdoc/AVID/AV23/AT100/full_res/AT100_100/heat_map/hm_map_0.1/heat_map_0.1_res10.nii
outp=$reg_dir/'ants_syn_AT8_'$id'_'
echo $outp

antsApplyTransforms -v -d 2 -i $mov -r $ref -n NearestNeighbor -t ${outp}1Warp.nii.gz -t ${outp}0GenericAffine.mat -o $out_img

echo $ref
echo $out
