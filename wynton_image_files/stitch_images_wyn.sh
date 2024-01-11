#!/bin/bash -l

if [ "$#" -ne 2 ]; then
	echo "Usage: stitch_image_wyn.sh <IMAGE_ROOT_DIR> <SLICE_NAME>"
	exit 0  
fi

#STITCHER_PATH='/home/maryana/bin/TeraStitcher/bin'

ROOT_DIR=$1
SLICE_NAME=$2
RAW_DIR=$ROOT_DIR'/raw'
OUT_DIR=$ROOT_DIR'/output'
TMP_DIR=$OUT_DIR'/magick_tmp'

#echo $ROOT_DIR
#echo $RAW_DIR
#echo $OUT_DIR

if [ -d "$OUT_DIR" ]; then
	rm -rf "$OUT_DIR"
fi
mkdir "$OUT_DIR"

if [ -d "$TMP_DIR" ]; then
        rm -rf "$TMP_DIR"
fi


cd $RAW_DIR
echo $PWD

echo 'Export XML'
#python /Users/prabhleenkaur/Box/Imaging_RO1_project/Wynton/LargeSlideScan/create_terastitch_xml.py $ROOT_DIR'/Metadata.txt' $RAW_DIR'/xml_import.xml' '/data/'$SLICE_NAME'/raw'
# Use below version for when running on local machine!
#python /Users/prabhleenkaur/Box/Imaging_RO1_project/Wynton/LargeSlideScan/create_terastitch_xml.py $ROOT_DIR'/Metadata.txt' $RAW_DIR'/xml_import.xml' $RAW_DIR
# Use Below version for when creating the image!
#python /root/create_terastitch_xml.py $ROOT_DIR'/Metadata.txt' $RAW_DIR'/xml_import.xml' $RAW_DIR
python2 create_terastitch_xml.py $ROOT_DIR'/Metadata.txt' $RAW_DIR'/xml_import.xml' $RAW_DIR
touch $ROOT_DIR'/expxml'

echo 'Import'
time terastitcher --import --projin="xml_import.xml" --imin_channel="G" 
touch $ROOT_DIR'/import'

echo 'Compute displacement'
time terastitcher --displcompute --projin="xml_import.xml" --imin_channel="G" #--noprogressbar 
touch $ROOT_DIR'/displ'

echo 'Compute projection'
time terastitcher --displproj --projin="xml_displcomp.xml" --imin_channel="G" 
touch $ROOT_DIR'/proj'

echo 'Threshold adj'
time terastitcher --displthres --threshold=0.7 --projin="xml_displproj.xml" --imin_channel="G" 
touch $ROOT_DIR'/thres'

echo 'Place tiles'
time terastitcher --placetiles --projin="xml_displthres.xml" --imin_channel="G"
touch $ROOT_DIR'/place'

echo 'Merge'
time teraconverter -s="xml_merging.xml" -d="$OUT_DIR" --sfmt="TIFF (unstitched, 3D)" --dfmt="TIFF (series, 2D)" --libtiff_bigtiff --noprogressbar
touch $ROOT_DIR'/merge'

#use Image Magick to create 10% res image

f=`ls $OUT_DIR'/'`

if [ -z "$f" ]
then
	echo 'Image not created. Nothing to do.'
else
	echo 'Resizing image.'

	mkdir "$TMP_DIR"

	export MAGICK_TMPDIR=$TMP_DIR
	export MAGICK_MEMORY_LIMIT=64Gb

	IMG_DIR=$OUT_DIR'/'$f
	IMG_DIR_ESC="$(echo "$IMG_DIR" | sed -e 's/[()&]/\\&/g')"
	cd $IMG_DIR
	IMG=`ls *.tif`
	IMG_OUT='res10_'$IMG

	convert -resize "10%" $IMG $IMG_OUT
	rm -rf "$TMP_DIR"
fi
