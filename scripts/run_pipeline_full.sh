#!/bin/bash -l

#export PYTHONPATH=$PYTHONPATH:/usr/local/bin/high-res-3D-tau

#if [ "$#" -ne 2 ]; then
#    echo "Usage: run_pipeline_full.sh <ROOT_DIR> <CONFIG_PATH>"
#    exit 0  
#fi

#ROOT_DIR=$1
#CONF_FILE=$2

#echo $ROOT_DIR
#echo $CONF_FILE

#python3 /usr/local/bin/high-res-3D-tau/pipeline/run_pipeline.py $ROOT_DIR $CONF_FILE

echo "--- Environment Debugging with --cleanenv ---"

echo "[INFO] PATH: $PATH"
echo "[INFO] PYTHONPATH: $PYTHONPATH"

echo ""
echo "--- Python Investigation ---"
PY_EXECUTABLE=$(command -v python3)

if [ -z "$PY_EXECUTABLE" ]; then
    echo "[ERROR] python3 not found in PATH!"
    exit 1
fi

echo "[INFO] 'command -v python3' found: $PY_EXECUTABLE"
echo "[INFO] Details of python3 executable:"
ls -l "$PY_EXECUTABLE"

echo ""
echo "--- Python's Own Path ---"
"$PY_EXECUTABLE" -c "import sys; print('Python sys.path:'); print('\n'.join(sys.path))"

echo ""
echo "--- Final Import Attempt ---"
"$PY_EXECUTABLE" -c "import skimage; print('Successfully imported skimage')"

echo "--- Debugging Finished ---"
