#!/bin/bash
#  Author: Michael Camilleri
#
#  Scope:
#     Runs inference on the Predict Set (folds) on MPCWork
#
#  Script takes the following parameter
#     [CLIP_LEN]     - Clip Length for AVA Sampler
#     [STRIDE]       - Inter-Frame Sampling Interval
#
#     [FOLD_NUM] - Fold Number
#     [BEST_MDL] - Name for best model
#
#  USAGE:
#     bash/predict_lfb.sh 9 49 &> predict_lfb.9.out
#     * N.B.: The above should be run from the root MMAction2 directory.

####  Some Configurations
# Get and store the main Parameters
CLIP_LEN=${1}
STRIDE=${2}

FOLD_NUM=${3}
BEST_MDL=${4}

# ===================
# Environment setup
# ===================
set -e # Make script bail out after first error
source activate py3mma   # Activate Conda Environment
echo "Libraries from: ${LD_LIBRARY_PATH}"

# Setup NCCL Debug Status
export NCCL_DEBUG=INFO

# Constants
FRAMES_DIR="../../Frames_Raw_Ext"
FEATURE_MDL="/media/veracrypt5/MRC_Data/Models/LFB/Base/feature_bank.base.pth"

# Scratch Space
SCRATCH_OUT="/media/veracrypt5/Scratch/LFB/out/${FOLD_NUM}"

# Variables
MODEL_FILE="/media/veracrypt5/MRC_Data/Models/LFB/Trained/Folds/${FOLD_NUM}/epoch_${BEST_MDL}.pth"
DATA_DIR="/media/veracrypt4/Q1/Snippets/Curated/Behaviour/Train/Folds/${FOLD_NUM}"

# Create Folders
mkdir -p ${SCRATCH_OUT}

# ================================
# Modify Configs
# ================================
echo " ===================================="
echo " Synchronising and Formatting Configs .. "
cp ${HOME}/Documents/Code/MMAction2/configs/own/backbone.base.py ${SCRATCH_OUT}/backbone.base.py
#  Update Feature-Bank Config
cp ${HOME}/Documents/Code/MMAction2/configs/own/feature_bank.base.py ${SCRATCH_OUT}/feature_bank.eval.py
sed -i "s@<SOURCE>@${DATA_DIR}@" ${SCRATCH_OUT}/feature_bank.eval.py
sed -i "s@<OUTPUT>@${SCRATCH_OUT}@" ${SCRATCH_OUT}/feature_bank.eval.py
sed -i "s@<DATASET>@Predict@" ${SCRATCH_OUT}/feature_bank.eval.py
sed -i "s@<FRAMES>@${FRAMES_DIR}@" ${SCRATCH_OUT}/feature_bank.eval.py
sed -i "s@<IMAGE_TEMPLATE>@img_{:05d}.jpg@" ${SCRATCH_MODELS}/feature_bank.eval.py
#  Update Inference Config
cp ${HOME}/Documents/Code/MMAction2/configs/own/infer.base.py ${SCRATCH_OUT}/infer.py
sed -i "s@<SOURCE>@${DATA_DIR}@" ${SCRATCH_OUT}/infer.py
sed -i "s@<FEATUREBANK>@${SCRATCH_OUT}@" ${SCRATCH_OUT}/infer.py
sed -i "s@<RESULTS>@${SCRATCH_OUT}@" ${SCRATCH_OUT}/infer.py
sed -i "s@<DATASET>@Predict@" ${SCRATCH_OUT}/infer.py
sed -i "s@<FRAMES>@${FRAMES_DIR}@" ${SCRATCH_OUT}/infer.py
sed -i "s@<IMAGE_TEMPLATE>@img_{:05d}.jpg@" ${SCRATCH_MODELS}/infer.py
sed -i "s@<CLEN>@${CLIP_LEN}@" ${SCRATCH_MODELS}/infer.py
sed -i "s@<STRIDE>@${STRIDE}@" ${SCRATCH_MODELS}/infer.py

# ======================
# Generate Feature Banks
# ======================
echo " ===================================="
echo "  Generating FB"
python tools/test.py \
      ${SCRATCH_OUT}/feature_bank.eval.py ${FEATURE_MDL} --out ${SCRATCH_OUT}/fb.csv --cfg-options data.test.start_index=125
rm -rf "${SCRATCH_OUT}/_lfb_*"
rm -rf "${SCRATCH_OUT}/*.csv"

# ================
# Infer Behaviours
# ================
echo " ===================================="
echo " Inferring Behaviours"
python tools/test.py \
    "${SCRATCH_OUT}"/infer.py "${MODEL_FILE}" --out "${SCRATCH_OUT}/Predict.csv" --cfg-options data.test.start_index=125
echo "   == Inference Done =="