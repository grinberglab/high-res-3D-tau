#!/bin/bash -l

if [ "$#" -ne 2 ]; then
        echo "Usage: create_dir_structure.sh <ROOT_DIR> <FOLDER_ID>"
        exit 0
fi

ROOT_DIR=$1
DIR_ID=$2

echo "${ROOT_DIR}"
echo "${DIR_ID}"

dir_root="${ROOT_DIR}"'/'$DIR_ID

output="${ROOT_DIR}"'/'$DIR_ID'/ouput'
res="${output}"'/RES'\('0x0'\)
img_tiles="${res}"'/tiles'

mask="${ROOT_DIR}"'/'$DIR_ID'/mask'
final_mask="${mask}"'/final_mask'
mask_tiles="${final_mask}"'/tiles'

heat_map="${ROOT_DIR}"'/'$DIR_ID'/heat_map'
cmap="${heat_map}"'/color_map_0.1'
hm_map="${heat_map}"'/hm_map_0.1'
seg_tiles="${heat_map}"'/seg_tiles'
TAU_tiles="${heat_map}"'/TAU_seg_tiles'

mkdir "${dir_root}"

mkdir "${output}"
mkdir "${res}"
mkdir "${img_tiles}"

mkdir "${mask}"
mkdir "${final_mask}"
mkdir "${mask_tiles}"

mkdir "${heat_map}"
mkdir "${cmap}"
mkdir "${hm_map}"
mkdir "${seg_tiles}"
mkdir "${TAU_tiles}"

