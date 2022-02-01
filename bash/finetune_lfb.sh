#!/bin/bash
#  Author: Michael Camilleri
#
#  Scope:
#     Fine-Tunes the LFB model (no explicit LFB/Backbone training beyond what goes on in backprop)
#
#  Script takes the following parameters
#     [Cores] - Number of GPUs to use to Train Model
#     [Images] - Number of Images per-GPU
#     [Epochs] - Maximum Number of Epochs to train for
#     [Rate] - Base Learning Rate (per sample, to be multiplied by batch size)
#
#  USAGE:
#     (Should be run from MMAction2 Directory)
#     srun --time=08:00:00 --gres=gpu:4 bash/finetune_lfb.sh C 4 4 20 # If on Charles nodes
#
#  Data Structures
#    Data is expected to be under ${HOME}/data/behaviour/[DATASET] where [DATASET]=Train/Validate
#    Model PTHs are under ${HOME}/models/LFB/Base/ : Configs are part of the Repository

# Do some Calculations
let "BATCH_SIZE=$1 * $2"
let "LEARN_RATE=$BATCH_SIZE * $4"

# ===================
# Environment setup
# ===================
echo "Setting up Conda enviroment on ${SLURM_JOB_NODELIST} with ${1} GPU(s)"
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
DATA_HOME=${SCRATCH_HOME}/data/behaviour
echo "  -> Synchronising Data"
mkdir -p ${DATA_HOME}
echo "    .. Training Set .. "
rsync --archive --update --compress ${HOME}/data/behaviour/Train ${DATA_HOME}/
echo "    .. Validation Set .. "
rsync --archive --update --compress ${HOME}/data/behaviour/Validate ${DATA_HOME}/
echo "    Data Done!"
echo " ------------------------------"
echo "  -> Synchronising Models"
MODEL_HOME=${SCRATCH_HOME}/models/lfb
echo "   .. Copying Models .. "
rsync --archive --update --compress ${HOME}/models/LFB/Base/ ${MODEL_HOME}/
echo "   .. Synchronising and Formatting Configs .. "
rsync --archive --update --compress ${HOME}/code/MMAction/configs/own/ ${MODEL_HOME}/
#  Update General FB Config
sed -i "s@<SOURCE>@${DATA_HOME}@" ${MODEL_HOME}/feature_bank.base.py
sed -i "s@<OUTPUT>@${DATA_HOME}/feature_bank@" ${MODEL_HOME}/feature_bank.base.py
#  Update V-Specific FB Config
cp ${MODEL_HOME}/feature_bank.base.py ${MODEL_HOME}/feature_bank.base.train.py
sed -i "s@<DATASET>@Train@" ${SCRATCH_HOME}/models/lfb/feature_bank.base.train.py
#  Update T-Specific FB Config
cp ${MODEL_HOME}/feature_bank.base.py ${MODEL_HOME}/feature_bank.base.valid.py
sed -i "s@<DATASET>@Validate@" ${SCRATCH_HOME}/models/lfb/feature_bank.base.valid.py
#  Update Training FB Config
sed -i "s@<SOURCE>@${DATA_HOME}@" ${MODEL_HOME}/train.base.py
sed -i "s@<FEATUREBANK>@${DATA_HOME}/feature_bank@" ${MODEL_HOME}/train.base.py
sed -i "s@<MODELINIT>@${MODEL_HOME}/inference.base.pth@" ${MODEL_HOME}/train.base.py
sed -i "s@<MODELOUT>@${MODEL_HOME}/out@" ${MODEL_HOME}/train.base.py
mkdir -p ${MODEL_HOME}/out
echo "    Models Done!"
mail -s "Train_LFB:Progress" ${USER}@sms.ed.ac.uk <<< "Synchronised Data and Models"
echo ""

# ======================
# Generate Feature Banks
#   Note that this is smart to not regenerate if the FB exists
# ======================
echo " ===================================="
echo " Generating Feature-Bank Vectors "
echo "  -> Training Set"
if [ -f "${DATA_HOME}/feature_bank/lfb_Train.pkl" ]; then
    echo "    Training Feature Bank exists: Skipping"
else
    python tools/test.py \
        ${SCRATCH_HOME}/models/lfb/feature_bank.base.train.py \
        ${SCRATCH_HOME}/models/lfb/feature_bank.base.pth \
        --out ${DATA_HOME}/feature_bank/train.csv
    echo "    Training FB Done"
fi
echo " ------------------------------"
echo "  -> Validation Set"
if [ -f "${DATA_HOME}/feature_bank/lfb_Validate.pkl" ]; then
    echo "    Validation Feature Bank exists: Skipping"
else
    python tools/test.py \
        ${SCRATCH_HOME}/models/lfb/feature_bank.base.valid.py \
        ${SCRATCH_HOME}/models/lfb/feature_bank.base.pth \
        --out ${DATA_HOME}/feature_bank/validate.csv
    echo "    Validation FB Done"
fi
echo " ------------------------------"
echo "  -> Cleaning up"
rm -rf ${DATA_HOME}/feature_bank/_lfb_*
rm -rf ${DATA_HOME}/feature_bank/*.csv
echo " == All Done =="
mail -s "Train_LFB:Progress" ${USER}@sms.ed.ac.uk <<< "Generated Feature Banks"
echo ""

# ===========
# Train Model
# ===========
echo " ===================================="
echo " Training Model with ${1} GPU(s)  (BS=${BATCH_SIZE}, LR=${LEARN_RATE}) for ${3} epochs"
python -m torch.distributed.launch --nproc_per_node=${1} tools/train.py \
    ${MODEL_HOME}/train.base.py --launcher pytorch \
    --validate --seed 0 --deterministic \
    --cfg-options data.videos_per_gpu=${2} total_epochs=${3} optimizer.lr=${LEARN_RATE}
mail -s "Train_LFB:Progress" ${USER}@sms.ed.ac.uk <<< "Model Training Completed."
echo ""

# ===========
# Copy Data
# ===========
echo " ===================================="
OUT_NAME = ${3}_${BATCH_SIZE}_${LEARN_RATE}
echo " Copying Results to ${OUT_NAME}"
mkdir -p "${HOME}/models/LFB/Trained/${OUT_NAME}"
rsync --archive --update --compress "${MODEL_HOME}/out/" "${HOME}/models/LFB/Trained/${OUT_NAME}"
rm -rf ${MODEL_HOME}/out
mail -s "Train_LFB:Progress" ${USER}@sms.ed.ac.uk <<< "Output Models copied."
conda deactivate