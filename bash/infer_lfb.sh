#!/bin/bash
#  Author: Michael Camilleri
#
#  Scope:
#     Runs inference on the Test-Set
#
#  Script takes the following parameter
#     [Model]    - Path (relative to ${HOME}/models/LFB/Trained) for the Model Weights
#     [Which]    - Which DataSet to evaluate (Train/Validate/Predict/Test)
#     [Offset]   - Offset from base data location to retrieve the data splits
#     [Frames]   - Y/N: Indicates if Frames should be rsynced: this is done to save time
#                       if it is known that the machine contains the right data splits.
#     [Features] - Y/N: If Y, force regenerate feature-banks.
#
#  USAGE:
#     srun --time=23:00:00 --gres=gpu:1 --nodelist=charles01 bash/infer_lfb.sh Fixed/50_16_0.000001_S/epoch_39.pth Validate Y Y &> ~/logs/infer_lfb.000001.out
#     * N.B.: The above should be run from the root MMAction2 directory. If need be, you can specify which machine to
#             run on explicitly through the --nodelist=charles<XX> argument
#
#  Data Structures
#    Data is expected to be under ${HOME}/data/behaviour/ which follows the definitions laid out
#        in my Jupyter notebook.
#    Model PTHs are under ${HOME}/models/LFB/Trained/: Configs are part of the Repository

####  Some Configurations
# Get and store the main Parameters
MODEL_PATH=${1}
DATASET=${2}
OFFSET=${3}
COPY_FRAMES=${4,,}
FORCE_LFB=${5,,}
# Derivative Values
CONFIG_PATH=$(dirname "${MODEL_PATH}")
CONFIG_NAME=$(basename "${CONFIG_PATH}")
if [ "${DATASET,,}" = "test" ]; then
  PARENT_DIR='Test'
else
  PARENT_DIR='Train'
fi

# ===================
# Environment setup
# ===================
echo "Setting up Conda enviroment on ${SLURM_JOB_NODELIST}"
set -e # Make script bail out after first error
source activate py3mma   # Activate Conda Environment
echo "Libraries from: ${LD_LIBRARY_PATH}"

# Setup NCCL Debug Status
export NCCL_DEBUG=INFO

# Make your own folder on the node's scratch disk
SCRATCH_HOME=/disk/scratch/${USER}
mkdir -p ${SCRATCH_HOME}
echo ""

# ================================
# Download Data and Models if necessary
# ================================
echo " ===================================="
echo "Consolidating Data/Models in ${SCRATCH_HOME}"
echo "  -> Synchronising Data"
SCRATCH_DATA=${SCRATCH_HOME}/data/behaviour
mkdir -p ${SCRATCH_DATA}
echo "     .. Schemas .."
cp ${HOME}/data/behaviour/Common/AVA* ${SCRATCH_DATA}/
echo "     .. Annotations .."
rsync --archive --update --compress --include '*/' --include 'AVA*' --exclude '*' \
      --info=progress2 ${HOME}/data/behaviour/${PARENT_DIR}/${OFFSET}/${DATASET} ${SCRATCH_DATA}
if [ "${COPY_FRAMES}" = "y" ]; then
  echo "     .. Frames .."
  rsync --archive --update --info=progress2 ${HOME}/data/behaviour/${PARENT_DIR}/Frames ${SCRATCH_DATA}
else
  echo "     .. Skipping Frames .."
fi
echo " ------------------------------"
echo "  -> Synchronising Models"
SCRATCH_MODELS=${SCRATCH_HOME}/models/lfb
echo "   .. Copying Models .. "
mkdir -p ${SCRATCH_MODELS}
# Copy the FB Inference Model and the Training Model (separately)
rsync --archive --compress ${HOME}/models/LFB/Base/feature_bank.base.pth ${SCRATCH_MODELS}/feature_bank.base.pth
rsync --archive --compress ${HOME}/models/LFB/Trained/${MODEL_PATH} ${SCRATCH_MODELS}/inference.trained.pth
echo "   .. Synchronising and Formatting Configs .. "
cp ${HOME}/code/MMAction/configs/own/backbone.base.py ${SCRATCH_MODELS}/backbone.base.py
#  Update Feature-Bank Config
cp ${HOME}/code/MMAction/configs/own/feature_bank.base.py ${SCRATCH_MODELS}/feature_bank.eval.py
sed -i "s@<SOURCE>@${SCRATCH_DATA}@" ${SCRATCH_MODELS}/feature_bank.eval.py
sed -i "s@<OUTPUT>@${SCRATCH_DATA}/feature_bank@" ${SCRATCH_MODELS}/feature_bank.eval.py
sed -i "s@<DATASET>@${DATASET}@" ${SCRATCH_HOME}/models/lfb/feature_bank.eval.py
#  Update Inference Config
cp ${HOME}/code/MMAction/configs/own/infer.base.py ${SCRATCH_MODELS}/infer.py
sed -i "s@<SOURCE>@${SCRATCH_DATA}@" ${SCRATCH_MODELS}/infer.py
sed -i "s@<FEATUREBANK>@${SCRATCH_DATA}/feature_bank@" ${SCRATCH_MODELS}/infer.py
sed -i "s@<RESULTS>@${SCRATCH_DATA}/out@" ${SCRATCH_MODELS}/infer.py
sed -i "s@<DATASET>@${DATASET}@" ${SCRATCH_MODELS}/infer.py
mkdir -p ${SCRATCH_DATA}/out
echo "    == Models Done =="
mail -s "Infer_LFB for ${DATASET} on ${SLURM_JOB_NODELIST}:${CONFIG_NAME}" ${USER}@sms.ed.ac.uk <<< "Synchronised Data and Models."
echo ""

# ======================
# Generate Feature Banks
# ======================
echo " ===================================="
if [ -f "${SCRATCH_DATA}/feature_bank/lfb_${DATASET}.pkl" ] && [ "${FORCE_LFB}" = "n" ]; then
  echo "    ${DATASET} FB Exists and not Forced to regenerate: skipping."
else
  echo "    Re-Generating"
  python tools/test.py \
      ${SCRATCH_MODELS}/feature_bank.eval.py \
      ${SCRATCH_MODELS}/feature_bank.base.pth \
      --out ${SCRATCH_DATA}/feature_bank/eval.csv
  echo "    == ${DATASET} FB Done =="
fi
echo " ------------------------------"
echo "  -> Cleaning up"
rm -rf ${SCRATCH_DATA}/feature_bank/_lfb_*
rm -rf ${SCRATCH_DATA}/feature_bank/*.csv
echo "  == FB Done =="
mail -s "Infer_LFB for ${DATASET} on ${SLURM_JOB_NODELIST}:${1}" ${USER}@sms.ed.ac.uk <<< "Generated Feature Banks"
echo ""

# ================
# Infer Behaviours
# ================
echo " ===================================="
echo " Inferring Behaviours for ${DATASET} using model ${MODEL_PATH}"
mkdir -p "${SCRATCH_DATA}/out"
python tools/test.py \
    ${SCRATCH_HOME}/models/lfb/infer.py \
    ${SCRATCH_MODELS}/inference.trained.pth \
    --out ${SCRATCH_DATA}/out/${DATASET}.csv
echo "   == Inference Done =="
mail -s "Infer_LFB for ${DATASET} on ${SLURM_JOB_NODELIST}:${CONFIG_NAME}" ${USER}@sms.ed.ac.uk <<< "Behaviour Inference Completed."
echo ""

# ===========
# Copy Data
# ===========
echo " ===================================="
RESULT_PATH="${HOME}/results/LFB/${CONFIG_PATH}"
echo " Copying Results to ${RESULT_PATH}"
mkdir -p ${RESULT_PATH}
rsync --archive --compress "${SCRATCH_DATA}/out/" ${RESULT_PATH}/
echo " Copying also LFB Features for posterity"
rsync --archive --compress "${SCRATCH_DATA}/feature_bank/lfb_${DATASET}.pkl" ${RESULT_PATH}/
rm -rf ${SCRATCH_DATA}/out
echo "   ++ ALL DONE! Hurray! ++"
mail -s "Infer_LFB for ${DATASET} on ${SLURM_JOB_NODELIST}:${CONFIG_NAME}" ${USER}@sms.ed.ac.uk <<< "Outputs copied to '${HOME}/results/LFB/${CONFIG_NAME}'."
conda deactivate