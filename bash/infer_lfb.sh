#!/bin/bash
#  Author: Michael Camilleri
#
#  Scope:
#     Runs inference on the Test-Set
#
#  Script takes the following parameter
#     [Model] - Path (relative to ${HOME}/models/LFB/Trained) for the Model Weights
#     [Which] - Which DataSet to evaluate (Test/Validate)
#     [Copy Data] - Y/N: Indicates if data should be copied or not (saves time). In this case, it is highly recommended
#                        to set the machine (as per below)
#
#  USAGE:
#     srun --time=04:00:00 --gres=gpu:1 bash/infer_lfb.sh 100_16_0.0001/epoch_15.pth Validate N &> ~/logs/lfb.tst.out
#     * N.B.: The above should be run from the root MMAction2 directory. If need be, you can specify which machine to
#             run on explicitly through the --nodelist=charles<XX> argument
#
#  Data Structures
#    Data is expected to be under ${HOME}/data/behaviour/[DATASET] where [DATASET] is one of Test/Validate
#    Model PTHs are under ${HOME}/models/LFB/Trained/: Configs are part of the Repository

# Some Configs
CONFIG_NAME=$(dirname "${1}")

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
SCRATCH_DATA=${SCRATCH_HOME}/data/behaviour
echo "  -> Synchronising Data"
if [ "${3,,}" = "y" ]; then
  mkdir -p ${SCRATCH_DATA}
  rsync --archive --update --compress --info=progress2 ${HOME}/data/behaviour/Test ${SCRATCH_DATA}/
  echo "    == Data Copied =="
else
  echo "    == Data is assumed Synchronised =="
fi
echo " ------------------------------"

echo "  -> Synchronising Models"
SCRATCH_MODELS=${SCRATCH_HOME}/models/lfb
echo "   .. Copying Models .. "
mkdir -p ${SCRATCH_MODELS}
# Copy the FB Inference Model and the Training Model (separately)
rsync --archive --update --compress ${HOME}/models/LFB/Base/feature_bank.base.pth ${SCRATCH_MODELS}/feature_bank.base.pth
rsync --archive --update --compress ${HOME}/models/LFB/Trained/${1} ${SCRATCH_MODELS}/inference.trained.pth
echo "   .. Synchronising and Formatting Configs .. "
rsync --archive --update --compress ${HOME}/code/MMAction/configs/own/ ${SCRATCH_MODELS}/
#  Update FB Config
cp ${SCRATCH_MODELS}/feature_bank.base.py ${SCRATCH_MODELS}/feature_bank.base.eval.py
sed -i "s@<SOURCE>@${SCRATCH_DATA}@" ${SCRATCH_MODELS}/feature_bank.eval.py
sed -i "s@<OUTPUT>@${SCRATCH_DATA}/feature_bank@" ${SCRATCH_MODELS}/feature_bank.eval.py
sed -i "s@<DATASET>@Test@" ${SCRATCH_HOME}/models/lfb/feature_bank.base.eval.py
#  Update Inference Config
sed -i "s@<SOURCE>@${SCRATCH_DATA}@" ${SCRATCH_MODELS}/infer.base.py
sed -i "s@<FEATUREBANK>@${SCRATCH_DATA}/feature_bank@" ${SCRATCH_MODELS}/infer.base.py
sed -i "s@<RESULTS>@${SCRATCH_DATA}/out@" ${SCRATCH_MODELS}/infer.base.py
mkdir -p ${SCRATCH_DATA}/out
echo "    == Models Done =="
mail -s "Infer_LFB for ${2} on ${SLURM_JOB_NODELIST}:${CONFIG_NAME}" ${USER}@sms.ed.ac.uk <<< "Synchronised Data and Models."
echo ""

# ======================
# Generate Feature Banks
#   Note that this is smart to not regenerate if the FB exists
# ======================
echo " ===================================="
echo " Generating Feature-Bank Vectors for ${2}"
if [ -f "${SCRATCH_DATA}/feature_bank/lfb_${2}.pkl" ]; then
    echo "    == ${2} FB Exists =="
else
    python tools/test.py \
        ${SCRATCH_MODELS}/feature_bank.base.eval.py \
        ${SCRATCH_MODELS}/feature_bank.base.pth \
        --out ${SCRATCH_DATA}/feature_bank/eval.csv
    echo "    == ${2} FB Done =="
fi
echo " ------------------------------"
echo "  -> Cleaning up"
rm -rf ${SCRATCH_DATA}/feature_bank/_lfb_*
rm -rf ${SCRATCH_DATA}/feature_bank/*.csv
echo "  == FB Done =="
mail -s "Infer_LFB for ${2} on ${SLURM_JOB_NODELIST}:${1}" ${USER}@sms.ed.ac.uk <<< "Generated Feature Banks"
echo ""

# ================
# Infer Behaviours
# ================
echo " ===================================="
echo " Inferring Behaviours for ${2}"
python tools/test.py \
    ${SCRATCH_HOME}/models/lfb/infer.base.py \
    ${SCRATCH_MODELS}/inference.trained.pth \
    --out ${SCRATCH_DATA}/out/${2}.csv
echo "   == Inference Done =="
mail -s "Infer_LFB for ${2} on ${SLURM_JOB_NODELIST}:${CONFIG_NAME}" ${USER}@sms.ed.ac.uk <<< "Behaviour Inferece Completed."
echo ""

# ===========
# Copy Data
# ===========
echo " ===================================="
echo " Copying Results to ${CONFIG_NAME}"
mkdir -p "${HOME}/results/LFB/${CONFIG_NAME}"
rsync --archive --update --compress "${SCRATCH_DATA}/out/" "${HOME}/results/LFB/${CONFIG_NAME}"
echo " Copying also LFB Features for posterity"
rsync --archive --update --compress "${SCRATCH_DATA}/feature_bank/lfb_${2}.pkl" "${HOME}/results/LFB/${CONFIG_NAME}/"
rm -rf ${SCRATCH_DATA}/out
echo "   ++ ALL DONE! Hurray! ++"
mail -s "Infer_LFB for ${2} on ${SLURM_JOB_NODELIST}:${CONFIG_NAME}" ${USER}@sms.ed.ac.uk <<< "Outputs copied to '${HOME}/results/LFB/${CONFIG_NAME}'."
conda deactivate