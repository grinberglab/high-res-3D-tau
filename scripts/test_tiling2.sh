#!/bin/bash -l

TMP_DIR=/data/magick_tmp

mkdir $TMP_DIR

export MAGICK_TMPDIR=$TMP_DIR
export MAGICK_MEMORY_LIMIT=64Gb
export MAGICK_MAP_LIMIT=64Gb

echo 'Converting image to mpc'
convert /data/000000_-85480_000000.tif /data/tmp.mpc

echo 'Tiling image'
convert -debug all /data/tmp.mpc -crop 18x20@ +repage +adjoin /data/tiles/tile_%04d.tif

echo 'Deleting temporary files'
rm /data/tmp.mpc
rm /data/tmp.cache
