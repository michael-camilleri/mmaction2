#!/bin/bash
#  Author: Michael Camilleri
#
#  Scope:
#     Runs inference on the Test-Set
#
#  Script takes the following parameter
#     [Model] - Path (relative to ${HOME}/models/LFB/Trained) for the Model Weights
#
#  USAGE:
#     srun --time=04:00:00 --gres=gpu:1 bash/infer_lfb.sh 100_16_0.0001/epoch_15.pth &> ~/logs/lfb.tst.out
#     * N.B.: The above should be run from the root MMAction2 directory. If need be, you can specify which machine to
#             run on explicitly through the --nodelist=charles<XX> argument
#
#  Data Structures
#    Data is expected to be under ${HOME}/data/behaviour/Test
#    Model PTHs are under ${HOME}/models/LFB/Trained/: Configs are part of the Repository

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
if [ "${5,,}" = "y" ]; then
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
rsync --archive --update --compress ${HOME}/models/LFB/Base/ ${SCRATCH_MODELS}/
rsync --archive --update --compress ${HOME}/models/LFB/Trained/${1} ${SCRATCH_MODELS}/
echo "   .. Synchronising and Formatting Configs .. "
rsync --archive --update --compress ${HOME}/code/MMAction/configs/own/ ${SCRATCH_MODELS}/
#  Update General FB Config
sed -i "s@<SOURCE>@${SCRATCH_DATA}@" ${SCRATCH_MODELS}/feature_bank.base.py
sed -i "s@<OUTPUT>@${SCRATCH_DATA}/feature_bank@" ${SCRATCH_MODELS}/feature_bank.base.py
#  Update V-Specific FB Config
cp ${SCRATCH_MODELS}/feature_bank.base.py ${SCRATCH_MODELS}/feature_bank.base.train.py
sed -i "s@<DATASET>@Train@" ${SCRATCH_HOME}/models/lfb/feature_bank.base.train.py
#  Update T-Specific FB Config
cp ${SCRATCH_MODELS}/feature_bank.base.py ${SCRATCH_MODELS}/feature_bank.base.valid.py
sed -i "s@<DATASET>@Validate@" ${SCRATCH_HOME}/models/lfb/feature_bank.base.valid.py
#  Update Training FB Config
sed -i "s@<SOURCE>@${SCRATCH_DATA}@" ${SCRATCH_MODELS}/train.base.py
sed -i "s@<FEATUREBANK>@${SCRATCH_DATA}/feature_bank@" ${SCRATCH_MODELS}/train.base.py
sed -i "s@<MODELINIT>@${SCRATCH_MODELS}/inference.base.pth@" ${SCRATCH_MODELS}/train.base.py
sed -i "s@<MODELOUT>@${SCRATCH_MODELS}/out@" ${SCRATCH_MODELS}/train.base.py
mkdir -p ${SCRATCH_MODELS}/out
echo "    == Models Done =="
mail -s "Train_LFB on ${SLURM_JOB_NODELIST}:${OUT_NAME}" ${USER}@sms.ed.ac.uk <<< "Synchronised Data and Models."
echo ""

# ======================
# Generate Feature Banks
#   Note that this is smart to not regenerate if the FB exists
# ======================
echo " ===================================="
echo " Generating Feature-Bank Vectors "
echo "  -> Training Set"
if [ -f "${SCRATCH_DATA}/feature_bank/lfb_Train.pkl" ]; then
    echo "    == Training FB Exists =="
else
    python tools/test.py \
        ${SCRATCH_HOME}/models/lfb/feature_bank.base.train.py \
        ${SCRATCH_HOME}/models/lfb/feature_bank.base.pth \
        --out ${SCRATCH_DATA}/feature_bank/train.csv
    echo "    == Training FB Done =="
fi
echo " ------------------------------"
echo "  -> Validation Set"
if [ -f "${SCRATCH_DATA}/feature_bank/lfb_Validate.pkl" ]; then
    echo "    == Validation FB Exists =="
else
    python tools/test.py \
        ${SCRATCH_HOME}/models/lfb/feature_bank.base.valid.py \
        ${SCRATCH_HOME}/models/lfb/feature_bank.base.pth \
        --out ${SCRATCH_DATA}/feature_bank/validate.csv
    echo "    == Validation FB Done =="
fi
echo " ------------------------------"
echo "  -> Cleaning up"
rm -rf ${SCRATCH_DATA}/feature_bank/_lfb_*
rm -rf ${SCRATCH_DATA}/feature_bank/*.csv
echo "  == FB Done =="
mail -s "Train_LFB on ${SLURM_JOB_NODELIST}:${OUT_NAME}" ${USER}@sms.ed.ac.uk <<< "Generated Feature Banks"
echo ""

# ===========
# Train Model
# ===========
echo " ===================================="
echo " Training Model with ${1} GPU(s)  (BS=${BATCH_SIZE}, LR=${LEARN_RATE}) for ${3} epochs"
python -m torch.distributed.launch --nproc_per_node=${1} tools/train.py \
    ${SCRATCH_MODELS}/train.base.py --launcher pytorch \
    --validate --seed 0 --deterministic \
    --cfg-options data.videos_per_gpu=${2} total_epochs=${3} optimizer.lr=${LEARN_RATE}
echo "   == Training Done =="
mail -s "Train_LFB on ${SLURM_JOB_NODELIST}:${OUT_NAME}" ${USER}@sms.ed.ac.uk <<< "Model Training Completed."
echo ""

# ===========
# Copy Data
# ===========
echo " ===================================="
echo " Copying Model Weights to ${OUT_NAME}"
mkdir -p "${HOME}/models/LFB/Trained/${OUT_NAME}"
rsync --archive --update --compress --info=progress2 "${SCRATCH_MODELS}/out/" "${HOME}/models/LFB/Trained/${OUT_NAME}"
echo " Copying also LFB Features"
rsync --archive --update --compress --info=progress2 "${SCRATCH_DATA}/feature_bank/" "${HOME}/models/LFB/Trained/${OUT_NAME}"
rm -rf ${SCRATCH_MODELS}/out
echo "   ++ ALL DONE! Hurray! ++"
mail -s "Train_LFB on ${SLURM_JOB_NODELIST}:${OUT_NAME}" ${USER}@sms.ed.ac.uk <<< "Output Models copied to '${HOME}/models/LFB/Trained/${OUT_NAME}'."
conda deactivate