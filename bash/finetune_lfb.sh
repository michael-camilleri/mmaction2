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
#     [Offset]   - Offset from base data location to retrieve the data splits
#     [Frames]   - Y/N: Indicates if Frames should be rsynced: this is done to save time if it is
#                       known that the machine contains the right data splits.
#     [Features] - Y/N: If Y, force regenerate feature-banks.
#
#  USAGE:
#     srun --time=1-23:00:00 --gres=gpu:8 --partition=apollo --nodelist=apollo1 bash/finetune_lfb.sh 8 2 0.00001 5 Fixed Y Y &> ~/logs/train_lfb.00001.Fixed.out
#     * N.B.: The above should be run from the root MMAction2 directory.

#  Data Structures
#    Data is expected to be under ${HOME}/data/behaviour/ which follows the definitions laid out
#        in my Jupyter notebook.
#    Model PTHs are under ${HOME}/models/LFB/Base/ : Configs are part of the Repository

# Do some Calculations/Preprocessing
BATCH_SIZE=$(echo "${1} * ${2}" | bc)
OUT_NAME=${4}_${BATCH_SIZE}_${3}_S10

# ===================
# Environment setup
# ===================
echo "Setting up Conda enviroment on ${SLURM_JOB_NODELIST} with ${1} GPU(s): Config=${OUT_NAME}"
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
SCRATCH_DATA=${SCRATCH_HOME}/data/behaviour/
mkdir -p ${SCRATCH_DATA}
echo "  -> Synchronising Data"
echo "     .. Schemas .."
cp ${HOME}/data/behaviour/Common/AVA* ${SCRATCH_DATA}/
echo "     .. Annotations .."
rsync --archive --update --compress --include '*/' --include 'AVA*' --exclude '*' \
      --info=progress2 ${HOME}/data/behaviour/Train/${5}/ ${SCRATCH_DATA}
if [ "${6,,}" = "y" ]; then
  echo "     .. Frames .."
  rsync --archive --update --info=progress2 ${HOME}/data/behaviour/Train/Frames ${SCRATCH_DATA}
else
  echo "     .. Skipping Frames .."
fi
echo " ------------------------------"
echo "  -> Synchronising Models"
SCRATCH_MODELS=${SCRATCH_HOME}/models/lfb_train
echo "   .. Copying Models .. "
mkdir -p ${SCRATCH_MODELS}
rsync --archive --update --compress ${HOME}/models/LFB/Base/ ${SCRATCH_MODELS}/
echo "   .. Synchronising and Formatting Configs .. "
cp ${HOME}/code/MMAction/configs/own/backbone.base.py ${SCRATCH_MODELS}/backbone.base.py
#  Update T-Specific FB Config
cp ${HOME}/code/MMAction/configs/own/feature_bank.base.py ${SCRATCH_MODELS}/feature_bank.train.py
sed -i "s@<SOURCE>@${SCRATCH_DATA}@" ${SCRATCH_MODELS}/feature_bank.train.py
sed -i "s@<OUTPUT>@${SCRATCH_DATA}/feature_bank@" ${SCRATCH_MODELS}/feature_bank.train.py
sed -i "s@<DATASET>@Train@" ${SCRATCH_HOME}/models/lfb/feature_bank.train.py
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
sed -i "s@<MODELOUT>@${SCRATCH_MODELS}/out@" ${SCRATCH_MODELS}/train.py
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
if [ -f "${SCRATCH_DATA}/feature_bank/lfb_Train.pkl" ] && [ "${7,,}" = "n" ]; then
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
if [ -f "${SCRATCH_DATA}/feature_bank/lfb_Validate.pkl" ] && [ "${7,,}" = "n" ]; then
  echo "    Validation FB Exists and not Forced to regenerate: skipping."
else
  echo "    Re-Generating"
  python tools/test.py \
      ${SCRATCH_MODELS}/feature_bank.valid.py \
      ${SCRATCH_MODELS}/feature_bank.base.pth \
      --out ${SCRATCH_DATA}/feature_bank/validate.csv
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
echo " Training Model with ${1} GPU(s)  (BS=${BATCH_SIZE}, LR=${3}) for ${4} epochs"
python -m torch.distributed.launch --nproc_per_node=${1} tools/train.py \
    ${SCRATCH_MODELS}/train.py --launcher pytorch \
    --validate --seed 0 --deterministic \
    --cfg-options data.videos_per_gpu=${2} optimizer.lr=${3} total_epochs=${4}
echo "   == Training Done =="
mail -s "Train_LFB on ${SLURM_JOB_NODELIST}:${OUT_NAME}" ${USER}@sms.ed.ac.uk <<< "Model Training Completed."
echo ""

# ===========
# Copy Data
# ===========
echo " ===================================="
OUTPUT_DIR="${HOME}/models/LFB/Trained/${5}"
mkdir -p ${OUTPUT_DIR}
echo " Copying Model Weights to ${OUTPUT_DIR}/${OUT_NAME}"
rsync --archive --compress --info=progress2 "${SCRATCH_MODELS}/out/" "${OUTPUT_DIR}/${OUT_NAME}"
echo " Copying also LFB Features"
rsync --archive --compress --info=progress2 "${SCRATCH_DATA}/feature_bank/" "${OUTPUT_DIR}/${OUT_NAME}"
rm -rf ${SCRATCH_MODELS}/out
echo "   ++ ALL DONE! Hurray! ++"
mail -s "Train_LFB on ${SLURM_JOB_NODELIST}:${OUT_NAME}" ${USER}@sms.ed.ac.uk <<< "Output Models copied to '${OUTPUT_DIR}/${OUT_NAME}'."
conda deactivate