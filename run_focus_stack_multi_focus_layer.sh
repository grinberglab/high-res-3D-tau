#!/bin/bash

for id in {0..623};
do echo $id;
cd '/home/maryana/R_DRIVE/Experiments/New_RO1/Scanned_images/Amyloid/1918/1918_22_#29_Amyloid_MFG_250ms/1918_22_#29_Amyloid_MFG_250ms_stack/1918_22_#29_Amyloid_MFG_250ms_stack_tiles_'$id;
align_image_stack -m -a OUT $(ls);
enfuse --compression=none --exposure-weight=0 --saturation-weight=0 --contrast-weight=1 --hard-mask --gray-projector=l-star --output='/home/maryana/R_DRIVE/Experiments/New_RO1/Scanned_images/Amyloid/1918/1918_22_#29_Amyloid_MFG_250ms/1918_22_#29_Amyloid_MFG_250ms_stack_tiles/raw/tile_'$id'.tif' OUT*.tif;
convert '/home/maryana/R_DRIVE/Experiments/New_RO1/Scanned_images/Amyloid/1918/1918_22_#29_Amyloid_MFG_250ms/1918_22_#29_Amyloid_MFG_250ms_stack_tiles/raw/tile_'$id'.tif' -alpha off '/home/maryana/R_DRIVE/Experiments/New_RO1/Scanned_images/Amyloid/1918/1918_22_#29_Amyloid_MFG_250ms/1918_22_#29_Amyloid_MFG_250ms_stack_tiles/raw/tile_'$id'.tif';
rm OUT*;
done

