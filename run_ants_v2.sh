#!/bin/bash

if [ "$#" -ne 1 ]; then
		echo "Usage: run_ants.sh <SLICE_ID>"
		exit 0
fi

id=$1
idblk=$(printf "%04d\n" $id)

ref='/Users/prabhleenkaur/Box/Imaging_RO1_project/Prabhleen/1918_BS/blockface_nii/1918_22_BS_w_cerebellum_0'$id'.nii'
mov='/Users/prabhleenkaur/Box/Imaging_RO1_project/Prabhleen/1918_BS/registration/'$id'/mipaved_1918_22_#'$id'_Iron_BS_300ms_focus_1_seg_regTPSpline.nii'
#mov=~/storage2/Posdoc/AVID/AV23/AT100/full_res/AT100/heat_map/hm_map_-71/heat_map_-7.1_res10.nii
outp='/Users/prabhleenkaur/Box/Imaging_RO1_project/Prabhleen/1918_BS/registration/'$id'/Step2_Automatic/1918_22_BS_'$id'_after_automatic_final_'
out='/Users/prabhleenkaur/Box/Imaging_RO1_project/Prabhleen/1918_BS/registration/'$id'/Step2_Automatic/1918_22_BS_'$id'_after_automatic_final.nii'

antsRegistration -v -d 2 -r [$ref,$mov,1] -m meansquares[$ref,$mov,1,32] -t rigid[0.1] -c 30x30x10 -s 4x2x1vox -f 3x3x1 -m mattes[$ref,$mov,1,32] -t Syn[0.5] -c 30x30x20 -s 3x3x1vox -f 4x2x2 -o [$outp]
antsApplyTransforms -v -d 2 -i $mov -r $ref -n linear -t ${outp}1Warp.nii.gz -t ${outp}0GenericAffine.mat -o $out

echo $ref
echo $out

