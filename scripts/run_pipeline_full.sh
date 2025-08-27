#!/bin/bash -l

#export PYTHONPATH=$PYTHONPATH:/usr/local/bin/high-res-3D-tau

# if [ "$#" -ne 2 ]; then
# 	echo "Usage: run_pipeline_full.sh <ROOT_DIR> <CONFIG_PATH>"
# 	exit 0  
# fi

# ROOT_DIR=$1
# CONF_FILE=$2

# echo $ROOT_DIR
# echo $CONF_FILE

# python3 /usr/local/bin/high-res-3D-tau/pipeline/run_pipeline.py $ROOT_DIR $CONF_FILE

# Diagnostic Test
#echo "Running diagnostic test..."
#python3 -c "import sys; import skimage; print('Python version:', sys.version); print('skimage version:', skimage.__version__)"
#echo "Diagnostic test finished."

echo "--- Environment Debugging ---"

echo "[INFO] Shell: $SHELL"
echo "[INFO] PATH: $PATH"
echo "[INFO] PYTHONPATH: $PYTHONPATH"

echo ""
echo "--- Python Investigation ---"
PY_EXECUTABLE=$(command -v python3)
echo "[INFO] 'command -v python3' found: $PY_EXECUTABLE"
echo "[INFO] Details of python3 executable:"
ls -l "$PY_EXECUTABLE"

echo ""
echo "--- Python's Own Path ---"
"$PY_EXECUTABLE" -c "import sys; print('\n'.join(sys.path))"

echo ""
echo "--- Final Import Attempt ---"
"$PY_EXECUTABLE" -c "import skimage; print('Successfully imported skimage')"

echo "--- Debugging Finished ---"
