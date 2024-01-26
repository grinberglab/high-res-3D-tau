#!/bin/bash

#make the folders

mkdir '/home/maryana/R_DRIVE/Experiments/New_RO1/Scanned_images/Amyloid/1918/1918_22_#29_Amyloid_MFG_250ms/1918_22_#29_Amyloid_MFG_250ms_stack' '/home/maryana/R_DRIVE/Experiments/New_RO1/Scanned_images/Amyloid/1918/1918_22_#29_Amyloid_MFG_250ms/1918_22_#29_Amyloid_MFG_250ms_stack_tiles' '/home/maryana/R_DRIVE/Experiments/New_RO1/Scanned_images/Amyloid/1918/1918_22_#29_Amyloid_MFG_250ms/1918_22_#29_Amyloid_MFG_250ms_stack_tiles/raw';

#copy the original metadata file to the new stacked_tiles folder

cp '/home/maryana/R_DRIVE/Experiments/New_RO1/Scanned_images/Amyloid/1918/1918_22_#29_Amyloid_MFG_250ms/1918_22_#29_Amyloid_MFG_250ms_focus_1/Metadata.txt' '/home/maryana/R_DRIVE/Experiments/New_RO1/Scanned_images/Amyloid/1918/1918_22_#29_Amyloid_MFG_250ms/1918_22_#29_Amyloid_MFG_250ms_stack_tiles/Metadata.txt';

#make new folders for each tiles that contain all the focus

for id in {0..623};
do echo $id;
mkdir '/home/maryana/R_DRIVE/Experiments/New_RO1/Scanned_images/Amyloid/1918/1918_22_#29_Amyloid_MFG_250ms/1918_22_#29_Amyloid_MFG_250ms_stack/1918_22_#29_Amyloid_MFG_250ms_stack_tiles_'$id;
#cd '/home/maryana/R_DRIVE/Experiments/1_End_of_Life/Scanned_Images/EOL-different versions scanned image for Jonathan/P2865_-3_26_focus_stack/P2865_-3_26_tiles_'$id'_stack';
mv '/home/maryana/R_DRIVE/Experiments/New_RO1/Scanned_images/Amyloid/1918/1918_22_#29_Amyloid_MFG_250ms/1918_22_#29_Amyloid_MFG_250ms_focus_1/raw/tile_'$id'.tif' '/home/maryana/R_DRIVE/Experiments/New_RO1/Scanned_images/Amyloid/1918/1918_22_#29_Amyloid_MFG_250ms/1918_22_#29_Amyloid_MFG_250ms_stack/1918_22_#29_Amyloid_MFG_250ms_stack_tiles_'$id'/tile_'$id'_1st.tif';

mv '/home/maryana/R_DRIVE/Experiments/New_RO1/Scanned_images/Amyloid/1918/1918_22_#29_Amyloid_MFG_250ms/1918_22_#29_Amyloid_MFG_250ms_focus_2/raw/tile_'$id'.tif' '/home/maryana/R_DRIVE/Experiments/New_RO1/Scanned_images/Amyloid/1918/1918_22_#29_Amyloid_MFG_250ms/1918_22_#29_Amyloid_MFG_250ms_stack/1918_22_#29_Amyloid_MFG_250ms_stack_tiles_'$id'/tile_'$id'_2nd.tif';


done

