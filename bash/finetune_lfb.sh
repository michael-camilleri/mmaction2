#!/bin/bash
#  Author: Michael Camilleri
#
#  Scope:
#     Fine-Tunes the LFB model (no explicit LFB/Backbone training beyond what goes on in backprop)
#
#  Script takes the following parameters: note that the batch size is defined by the product of
#    Cores and Images.
#     [Cores]    - Number of GPUs to use to Train Model
#     [Images]   - Number of Images (Samples) per-GPU
#     [Rate]     - Learning Rate (actual value used, not dependent on Batch-Size)
#     [Epochs]   - Maximum Number of Epochs to train for
#     [Warmup]   - Warmup Period (epochs)
#     [Offset]   - Offset from base data location to retrieve the data splits
#     [Frames]   - Y/N: Indicates if Frames should be rsynced: this is done to save time if it is
#                       known that the machine contains the right data splits.
#     [Features] - Y/N: If Y, force regenerate feature-banks.
#
#  USAGE:
#     srun --time=1-23:00:00 --gres=gpu:8 --partition=apollo --nodelist=apollo1 bash/finetune_lfb.sh 8 2 0.00001 50 10 Fixed Y Y &> ~/logs/train_lfb.00001.Fixed.out
#     * N.B.: The above should be run from the root MMAction2 directory.

#  Data Structures
#    Data is expected to be under ${HOME}/data/behaviour/ which follows the definitions laid out
#        in my Jupyter notebook.
#    Model PTHs are under ${HOME}/models/LFB/Base/ : Configs are part of the Repository

####  Some Configurations
# Get and store the main Parameters
GPU_NODES=${1}
IMAGE_GPU=${2}
LEARN_RATE=${3}
MAX_EPOCHS=${4}
WARMUP_ITER=${5}
PATH_OFFSET=${6}
FORCE_FRAMES=${7,,}
FORCE_LFB=${8,,}

# Derivative Values
BATCH_SIZE=$(echo "${GPU_NODES} * ${IMAGE_GPU}" | bc)
OUT_NAME=${MAX_EPOCHS}_${BATCH_SIZE}_L${LEARN_RATE}_W${WARMUP_ITER}_CJ

# Path Values
SCRATCH_HOME=/disk/scratch/${USER}
SCRATCH_DATA=${SCRATCH_HOME}/data/behaviour
SCRATCH_MODELS=${SCRATCH_HOME}/models/lfb_train
SCRATCH_OUT=${SCRATCH_HOME}/results_train
OUTPUT_DIR="${HOME}/models/LFB/Trained/${PATH_OFFSET}/${OUT_NAME}"

# ===================
# Environment setup
# ===================
echo "Setting up Conda enviroment on ${SLURM_JOB_NODELIST} with ${GPU_NODES} GPU(s): Config=${OUT_NAME}"
set -e # Make script bail out after first error
source activate py3mma   # Activate Conda Environment
echo "Libraries from: ${LD_LIBRARY_PATH}"

# Setup NCCL Debug Status
export NCCL_DEBUG=INFO

# Make your own folder on the node's scratch disk
mkdir -p "${SCRATCH_HOME}"
echo ""

# ================================
# Download Data and Models if necessary
# ================================
echo " ===================================="
echo "Consolidating Data/Models in ${SCRATCH_HOME}"
mkdir -p "${SCRATCH_DATA}"
echo "  -> Synchronising Data"
echo "     .. Schemas .."
cp ${HOME}/data/behaviour/Common/AVA* "${SCRATCH_DATA}/"
echo "     .. Annotations .."
rsync --archive --update --compress --include '*/' --include 'AVA*' --exclude '*' \
      --info=progress2 "${HOME}/data/behaviour/Train/${PATH_OFFSET}/" "${SCRATCH_DATA}/"
if [ "${FORCE_FRAMES}" = "y" ]; then
  echo "     .. Frames .."
  rsync --archive --update --info=progress2 "${HOME}/data/behaviour/Train/Frames" "${SCRATCH_DATA}/"
else
  echo "     .. Skipping Frames .."
fi
echo " ------------------------------"
echo "  -> Synchronising Models"
echo "   .. Copying Models .. "
mkdir -p ${SCRATCH_MODELS}
mkdir -p ${SCRATCH_OUT}
rsync --archive --update --compress ${HOME}/models/LFB/Base/ ${SCRATCH_MODELS}/
echo "   .. Synchronising and Formatting Configs .. "
cp ${HOME}/code/MMAction/configs/own/backbone.base.py ${SCRATCH_MODELS}/backbone.base.py
#  Update T-Specific FB Config
cp ${HOME}/code/MMAction/configs/own/feature_bank.base.py ${SCRATCH_MODELS}/feature_bank.train.py
sed -i "s@<SOURCE>@${SCRATCH_DATA}@" ${SCRATCH_MODELS}/feature_bank.train.py
sed -i "s@<OUTPUT>@${SCRATCH_DATA}/feature_bank@" ${SCRATCH_MODELS}/feature_bank.train.py
sed -i "s@<DATASET>@Train@" ${SCRATCH_MODELS}/feature_bank.train.py
#  Update V-Specific FB Config
cp ${HOME}/code/MMAction/configs/own/feature_bank.base.py ${SCRATCH_MODELS}/feature_bank.valid.py
sed -i "s@<SOURCE>@${SCRATCH_DATA}@" ${SCRATCH_MODELS}/feature_bank.valid.py
sed -i "s@<OUTPUT>@${SCRATCH_DATA}/feature_bank@" ${SCRATCH_MODELS}/feature_bank.valid.py
sed -i "s@<DATASET>@Validate@" ${SCRATCH_MODELS}/feature_bank.valid.py
#  Update Training FB Config (Now using only SGD)
cp ${HOME}/code/MMAction/configs/own/train_sgd.base.py ${SCRATCH_MODELS}/train.py
sed -i "s@<SOURCE>@${SCRATCH_DATA}@" ${SCRATCH_MODELS}/train.py
sed -i "s@<FEATUREBANK>@${SCRATCH_DATA}/feature_bank@" ${SCRATCH_MODELS}/train.py
sed -i "s@<MODELINIT>@${SCRATCH_MODELS}/inference.base.pth@" ${SCRATCH_MODELS}/train.py
sed -i "s@<MODELOUT>@${SCRATCH_OUT}@" ${SCRATCH_MODELS}/train.py
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
if [ -f "${SCRATCH_DATA}/feature_bank/lfb_Train.pkl" ] && [ "${FORCE_LFB}" = "n" ]; then
  echo "    Training FB Exists and not Forced to regenerate: skipping."
else
  echo "    Re-Generating"
  python tools/test.py \
      ${SCRATCH_MODELS}/feature_bank.train.py \
      ${SCRATCH_MODELS}/feature_bank.base.pth \
      --out ${SCRATCH_DATA}/feature_bank/train.csv
  echo "     Training FB Done"
fi
echo " ------------------------------"
echo "  -> Validation Set"
if [ -f "${SCRATCH_DATA}/feature_bank/lfb_Validate.pkl" ] && [ "${FORCE_LFB}" = "n" ]; then
  echo "    Validation FB Exists and not Forced to regenerate: skipping."
else
  echo "    Re-Generating"
  python tools/test.py \
      "${SCRATCH_MODELS}/feature_bank.valid.py" \
      "${SCRATCH_MODELS}/feature_bank.base.pth" \
      --out "${SCRATCH_DATA}/feature_bank/validate.csv"
  echo "     Validation FB Done"
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
echo " Training Model with ${GPU_NODES} GPU(s)  (BS=${BATCH_SIZE}, LR=${LEARN_RATE}) for ${MAX_EPOCHS} epochs"
python -m torch.distributed.launch --nproc_per_node="${GPU_NODES}" tools/train.py \
    "${SCRATCH_MODELS}/train.py" --launcher pytorch \
    --validate --seed 0 --deterministic \
    --cfg-options data.videos_per_gpu="${IMAGE_GPU}" optimizer.lr="${LEARN_RATE}" total_epochs="${MAX_EPOCHS}" lr_config.warmup_iters="${WARMUP_ITER}"
echo "   == Training Done =="
mail -s "Train_LFB on ${SLURM_JOB_NODELIST}:${OUT_NAME}" ${USER}@sms.ed.ac.uk <<< "Model Training Completed."
echo ""

# ===========
# Copy Data
# ===========
echo " ===================================="
mkdir -p ${OUTPUT_DIR}
echo " Copying Model Weights to ${OUTPUT_DIR}"
rsync --archive --compress --info=progress2 "${SCRATCH_MODELS}/out/" "${OUTPUT_DIR}"
echo " Copying also LFB Features"
rsync --archive --compress --info=progress2 "${SCRATCH_DATA}/feature_bank/" "${OUTPUT_DIR}"
rm -rf "${SCRATCH_MODELS}/out"
echo "   ++ ALL DONE! Hurray! ++"
mail -s "Train_LFB on ${SLURM_JOB_NODELIST}:${OUT_NAME}" ${USER}@sms.ed.ac.uk <<< "Output Models copied to '${OUTPUT_DIR}'."
conda deactivate