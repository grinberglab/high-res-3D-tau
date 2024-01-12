#!/bin/bash -l

if [ "$#" -ne 4 ]; then
	echo "Usage: stitch_image_wyn.sh <INPUT_FILE> <TILES> <TILES_DIR> <TMP_DIR>"
	exit 0
fi

INPUT_FILE=$1
TILE_STR=$2
TILES_DIR=$3
TMP_DIR=$4

echo '-Input file: '$INPUT_FILE
echo '-Tiles: '$TILE_STR
echo '-Tiles dir: '$TILES_DIR
echo '-Tmp dir :'$TMP_DIR

export MAGICK_TMPDIR=$TMP_DIR
export MAGICK_MEMORY_LIMIT=64Gb
export MAGICK_MAP_LIMIT=64Gb

echo '-Tiling image'
convert -debug all $INPUT_FILE -crop $TILE_STR +repage +adjoin $TILES_DIR/tile_%04d.tif
