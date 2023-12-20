#!/bin/bash -l

if [ "$#" -ne 1 ]; then
        echo "Usage: run_convert.sh <DIR>"
        exit 0
fi


DIR_IN=$1
DIR_OUT=$2
TMP_DIR=/grinberg/scratch/tmp

#create temp dir
#if [ -d "$TMP_DIR" ]; then
#        rm -rf "$TMP_DIR"
#fi
#mkdir "$TMP_DIR"

echo $DIR_IN

#resize image
f=`ls  $DIR_IN'/'` #should be /path/to/<case>/output

echo $f

if [ -z "$f" ]
then
        echo 'Image not created. Nothing to do.'
else
        echo 'Resizing image.'

        export MAGICK_TMPDIR=$TMP_DIR
        export MAGICK_MEMORY_LIMIT=64Gb

        IMG_DIR=$DIR_IN'/'$f
        IMG_DIR_ESC="$(echo "$IMG_DIR" | sed -e 's/[()&]/\\&/g')"
        cd $IMG_DIR
        IMG=`ls *.tif`
        IMG_OUT='res10_'$IMG

	echo $IMG
	echo $IMG_OUT

        convert -resize "10%" $IMG $IMG_OUT
fi
