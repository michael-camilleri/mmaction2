#!/bin/bash
#  Author: Michael Camilleri
#
#  Scope:
#     Fine-Tunes the LFB model (no explicit LFB/Backbone training beyond what goes on in backprop)
#
#  Script takes the following parameters: note that the batch size is defined by the product of
#    Cores and Images.
#     [CLIP_LEN]     - Clip Length for AVA Sampler
#     [STRIDE]       - Inter-Frame Sampling Interval

#     [GPU_NODES]    - Number of GPUs to use to Train Model
#     [IMAGE_GPU]    - Number of Images (Samples) per-GPU
#     [LEARN_RATE]   - Learning Rate (actual value used, not dependent on Batch-Size)
#     [MAX_EPOCHS]   - Maximum Number of Epochs to train for

#     [PATH_OFFSET]  - Offset from base data location to retrieve the data splits
#     [FRAMES_DIR]   - Frame Directory to use (offset from base location)
#     [FRAME_NUM]    - Starting Index for Frame Numbering
#     [FORCE_FRAMES] - Y/N: Indicates if Frames should be rsynced: this is done to save time if it
#                           is known that the machine contains the right data splits.
#     [FORCE_LFB]    - Y/N: If Y, force regenerate feature-banks.
#
#  USAGE:
#     srun --time=2-23:00:00 --gres=gpu:4 --partition=apollo --nodelist=apollo1 bash/finetune_lfb.sh 5 16 4 4 0.0005 50 Fixed Frames_Raw_Ext 125 N Y [29500] &> ~/logs/train_lfb.C5S16.out
#     * N.B.: The above should be run from the root MMAction2 directory.

#  Data Structures
#    Data is expected to be under ${HOME}/data/behaviour/ which follows the definitions laid out
#        in my Jupyter notebook.
#    Model PTHs are under ${HOME}/models/LFB/Base/ : Configs are part of the Repository

####  Some Configurations
# Get and store the main Parameters
CLIP_LEN=${1}
STRIDE=${2}

GPU_NODES=${3}
IMAGE_GPU=${4}
LEARN_RATE=${5}
MAX_EPOCHS=${6}

PATH_OFFSET=${7}
FRAMES_DIR=${8}
FRAME_NUM=${9}
FORCE_FRAMES=${10,,}
FORCE_LFB=${11,,}

PORT=${12:-29500}

# Derivative Values
BATCH_SIZE=$(echo "${GPU_NODES} * ${IMAGE_GPU}" | bc)
OUT_NAME=LFB_C${CLIP_LEN}_S${STRIDE}_L${LEARN_RATE}

# Path Values
SCRATCH_HOME=/disk/scratch/${USER}
SCRATCH_DATA=${SCRATCH_HOME}/data/behaviour
SCRATCH_MODELS=${SCRATCH_HOME}/models/lfb_train/${OUT_NAME}
SCRATCH_OUT=${SCRATCH_HOME}/results_train/${OUT_NAME}
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
rsync --archive --compress --include '*/' --include 'AVA*' --exclude '*' \
      --info=progress2 "${HOME}/data/behaviour/Train/${PATH_OFFSET}/" "${SCRATCH_DATA}/"
if [ "${FORCE_FRAMES}" = "y" ]; then
  echo "     .. Frames .."
  mkdir -p "${SCRATCH_DATA}/${FRAMES_DIR}"
  rsync --archive --update --info=progress2 "${HOME}/data/behaviour/Train/${FRAMES_DIR}" \
        "${SCRATCH_DATA}/"
else
  echo "     .. Skipping Frames .."
fi
echo " ------------------------------"
echo "  -> Synchronising Models"
echo "   .. Copying Models .. "
mkdir -p ${SCRATCH_MODELS}
mkdir -p ${SCRATCH_OUT}
rsync --archive --compress ${HOME}/models/LFB/Base/ ${SCRATCH_MODELS}/
echo "   .. Synchronising and Formatting Configs .. "
cp ${HOME}/code/MMAction/configs/own/backbone.base.py ${SCRATCH_MODELS}/backbone.base.py
#  Update T-Specific FB Config
cp ${HOME}/code/MMAction/configs/own/feature_bank.base.py ${SCRATCH_MODELS}/feature_bank.train.py
sed -i "s@<SOURCE>@${SCRATCH_DATA}@" ${SCRATCH_MODELS}/feature_bank.train.py
sed -i "s@<OUTPUT>@${SCRATCH_MODELS}/feature_bank@" ${SCRATCH_MODELS}/feature_bank.train.py
sed -i "s@<DATASET>@Train@" ${SCRATCH_MODELS}/feature_bank.train.py
sed -i "s@<FRAMES>@${FRAMES_DIR}@" ${SCRATCH_MODELS}/feature_bank.train.py
sed -i "s@<IMAGE_TEMPLATE>@img_{:05d}.jpg@" ${SCRATCH_MODELS}/feature_bank.train.py
#  Update V-Specific FB Config
cp ${HOME}/code/MMAction/configs/own/feature_bank.base.py ${SCRATCH_MODELS}/feature_bank.valid.py
sed -i "s@<SOURCE>@${SCRATCH_DATA}@" ${SCRATCH_MODELS}/feature_bank.valid.py
sed -i "s@<OUTPUT>@${SCRATCH_MODELS}/feature_bank@" ${SCRATCH_MODELS}/feature_bank.valid.py
sed -i "s@<DATASET>@Validate@" ${SCRATCH_MODELS}/feature_bank.valid.py
sed -i "s@<FRAMES>@${FRAMES_DIR}@" ${SCRATCH_MODELS}/feature_bank.valid.py
sed -i "s@<IMAGE_TEMPLATE>@img_{:05d}.jpg@" ${SCRATCH_MODELS}/feature_bank.valid.py
#  Update Training FB Config (Now using only SGD)
cp ${HOME}/code/MMAction/configs/own/train_sgd.base.py ${SCRATCH_MODELS}/train.py
sed -i "s@<SOURCE>@${SCRATCH_DATA}@" ${SCRATCH_MODELS}/train.py
sed -i "s@<FEATUREBANK>@${SCRATCH_MODELS}/feature_bank@" ${SCRATCH_MODELS}/train.py
sed -i "s@<MODELINIT>@${SCRATCH_MODELS}/inference.base.pth@" ${SCRATCH_MODELS}/train.py
sed -i "s@<MODELOUT>@${SCRATCH_OUT}@" ${SCRATCH_MODELS}/train.py
sed -i "s@<FRAMES>@${FRAMES_DIR}@" ${SCRATCH_MODELS}/train.py
sed -i "s@<IMAGE_TEMPLATE>@img_{:05d}.jpg@" ${SCRATCH_MODELS}/train.py
sed -i "s@<CLEN>@${CLIP_LEN}@" ${SCRATCH_MODELS}/train.py
sed -i "s@<STRIDE>@${STRIDE}@" ${SCRATCH_MODELS}/train.py
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
if [ -f "${SCRATCH_MODELS}/feature_bank/lfb_Train.pkl" ] && [ "${FORCE_LFB}" = "n" ]; then
  echo "    Training FB Exists and not Forced to regenerate: skipping."
else
  echo "    Re-Generating"
  python tools/test.py \
      ${SCRATCH_MODELS}/feature_bank.train.py \
      ${SCRATCH_MODELS}/feature_bank.base.pth \
      --out ${SCRATCH_MODELS}/feature_bank/train.csv \
      --cfg-options data.test.start_index="${FRAME_NUM}"
  echo "     Training FB Done"
fi
echo " ------------------------------"
echo "  -> Validation Set"
if [ -f "${SCRATCH_MODELS}/feature_bank/lfb_Validate.pkl" ] && [ "${FORCE_LFB}" = "n" ]; then
  echo "    Validation FB Exists and not Forced to regenerate: skipping."
else
  echo "    Re-Generating"
  python tools/test.py \
      "${SCRATCH_MODELS}/feature_bank.valid.py" \
      "${SCRATCH_MODELS}/feature_bank.base.pth" \
      --out "${SCRATCH_MODELS}/feature_bank/validate.csv" \
      --cfg-options data.test.start_index="${FRAME_NUM}"
  echo "     Validation FB Done"
fi
echo " ------------------------------"
echo "  -> Cleaning up"
rm -rf ${SCRATCH_MODELS}/feature_bank/_lfb_*
rm -rf ${SCRATCH_MODELS}/feature_bank/*.csv
echo "  == FB Done =="
mail -s "Train_LFB on ${SLURM_JOB_NODELIST}:${OUT_NAME}" ${USER}@sms.ed.ac.uk <<< "Generated Feature Banks"
echo ""

# ===========
# Train Model
# ===========
echo " ===================================="
echo " Training Model with ${GPU_NODES} GPU(s)  (BS=${BATCH_SIZE}, LR=${LEARN_RATE}) for ${MAX_EPOCHS} epochs"
python -m torch.distributed.launch --nproc_per_node="${GPU_NODES}" --master_port="${PORT}" tools/train.py \
    "${SCRATCH_MODELS}/train.py" --launcher pytorch \
    --validate --seed 0 --deterministic \
    --cfg-options data.videos_per_gpu="${IMAGE_GPU}" optimizer.lr="${LEARN_RATE}" total_epochs="${MAX_EPOCHS}" data.train.start_index="${FRAME_NUM}" data.val.start_index="${FRAME_NUM}"
echo "   == Training Done =="
mail -s "Train_LFB on ${SLURM_JOB_NODELIST}:${OUT_NAME}" ${USER}@sms.ed.ac.uk <<< "Model Training Completed."
echo ""

# ===========
# Copy Data
# ===========
echo " ===================================="
mkdir -p ${OUTPUT_DIR}
echo " Copying Model Weights to ${OUTPUT_DIR}"
rsync --archive --compress --info=progress2 "${SCRATCH_OUT}/" "${OUTPUT_DIR}"
echo " Copying also LFB Features"
rsync --archive --compress --info=progress2 "${SCRATCH_MODELS}/feature_bank/" "${OUTPUT_DIR}"
rm -rf "${SCRATCH_OUT}"
echo "   ++ ALL DONE! Hurray! ++"
mail -s "Train_LFB on ${SLURM_JOB_NODELIST}:${OUT_NAME}" ${USER}@sms.ed.ac.uk <<< "Output Models copied to '${OUTPUT_DIR}'."
conda deactivate