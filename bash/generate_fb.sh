#!/bin/bash
#  Author: Michael Camilleri
#
#  Scope:
#     Generates FB on some DataSet (to be run on MPCWork)
#
#  Script takes the following parameter
#     [PATH_OFFSET]  - Offset from base data location to retrieve the data splits
#     [DATASET]      - Which DataSet to evaluate (Train/Validate/Test)
#     [FRAMES]       - Frames directory (offset from PATH_OFFSET)
#
#  USAGE:
#     bash/generate_fb.sh End2End Train ../Frames
#     * N.B.: The above should be run from the root MMAction2 directory.

####  Some Configurations
# Get and store the main Parameters
PATH_OFFSET='/media/veracrypt4/Q1/Snippets/Curated/Behaviour/'${1}
DATASET=${2}
FRAMES=${3}

# ===================
# Environment setup
# ===================
set -e # Make script bail out after first error
source activate py3mma   # Activate Conda Environment
echo "Libraries from: ${LD_LIBRARY_PATH}"

# Setup some Config Options
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Setup NCCL Debug Status
export NCCL_DEBUG=INFO

# Constants
FEATURE_MDL="/media/veracrypt5/MRC_Data/Models/LFB/Base/feature_bank.base.pth"

# Scratch Space
SCRATCH_OUT="/home/s1238640/Documents/Data/scratch_fb_${DATASET}"
mkdir -p ${SCRATCH_OUT}

# ================================
# Modify Configs
# ================================
echo " ===================================="
echo " Synchronising and Formatting Configs .. "
cp ${HOME}/Documents/Code/MMAction2/configs/own/backbone.base.py ${SCRATCH_OUT}/backbone.base.py
#  Update Feature-Bank Config
cp ${HOME}/Documents/Code/MMAction2/configs/own/feature_bank.base.py ${SCRATCH_OUT}/feature_bank.eval.py
sed -i "s@<SOURCE>@${PATH_OFFSET}@" ${SCRATCH_OUT}/feature_bank.eval.py
sed -i "s@<OUTPUT>@${SCRATCH_OUT}@" ${SCRATCH_OUT}/feature_bank.eval.py
sed -i "s@<DATASET>@${DATASET}@" ${SCRATCH_OUT}/feature_bank.eval.py
sed -i "s@<FRAMES>@${FRAMES}@" ${SCRATCH_OUT}/feature_bank.eval.py
sed -i "s@<IMAGE_TEMPLATE>@img_{:05d}.jpg@" ${SCRATCH_OUT}/feature_bank.eval.py

# ======================
# Generate Feature Banks
# ======================
echo " ===================================="
echo "  Generating FB"
python tools/test.py \
      ${SCRATCH_OUT}/feature_bank.eval.py ${FEATURE_MDL} --out ${SCRATCH_OUT}/fb.csv --cfg-options data.test.start_index=0
rm -rf "${SCRATCH_OUT}/_lfb_*"
rm -rf "${SCRATCH_OUT}/*.csv"