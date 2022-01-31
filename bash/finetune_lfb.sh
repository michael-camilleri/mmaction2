#!/bin/bash
#  Author: Michael Camilleri
#
#  Scope:
#     Fine-Tunes the LFB model (no explicit LFB/Backbone training beyond what goes on in backprop)
#
#  Script takes two parameters
#     A/C - Choice between running on Apollo or Charles Nodes
#     [Cores] - Number of GPUs (parallel) to use
#
#  USAGE:
#     srun --time=08:00:00 --gres=gpu:1 finetune_lfb.sh C 1 # If on Charles nodes
#
#  Data Structures
#    Data is expected to be under ${HOME}/data/behaviour/[DATASET] where [DATASET]=Train/Validate
#    Model PTHs are under ${HOME}/models/LFB/Base/ : Configs are part of the Repository


# ===================
# Environment setup
# ===================
echo "Setting up Conda enviroment on ${SLURM_JOB_NODELIST} (${1}) with ${2} GPU(s)."
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
DATA_HOME=${SCRATCH_HOME}/data/behaviour/
echo "  -> Synchronising Data"
mkdir -p ${DATA_HOME}
#echo "    .. Train .. "
#rsync --archive --update --compress ${HOME}/data/behaviour/Train ${DATA_HOME}/
echo "    .. Validate .. "
rsync --archive --update --compress ${HOME}/data/behaviour/Validate ${DATA_HOME}/
echo "    Data Done!"
echo " ------------------------------"
echo "  -> Synchronising Models"
mkdir -p ${SCRATCH_HOME}/models/lfb/
rsync --archive --update --compress ${HOME}/models/LFB/Base/ ${SCRATCH_HOME}/models/lfb/
echo "   .. Synchronising and Formatting Configs .. "
rsync --archive --update --compress ${HOME}/conde/MMAction/configs/own/ ${SCRATCH_HOME}/models/lfb/
sed -i "s/# <SOURCE>/Source_Root=${DATA_HOME}/" ${SCRATCH_HOME}/models/lfb/feature_bank.base.blank.py
sed -i "s/# <OUTPUT>/Output_Path=${DATA_HOME}/" ${SCRATCH_HOME}/models/lfb/feature_bank.base.blank.py
cp ${SCRATCH_HOME}/models/lfb/feature_bank.base.blank.py ${SCRATCH_HOME}/models/lfb/feature_bank.base.train.py
sed -i "s/# <DATASET>/DataSet=Train/" ${SCRATCH_HOME}/models/lfb/feature_bank.base.train.py
cp ${SCRATCH_HOME}/models/lfb/feature_bank.base.blank.py ${SCRATCH_HOME}/models/lfb/feature_bank.base.valid.py
sed -i "s/# <DATASET>/DataSet=Validate/" ${SCRATCH_HOME}/models/lfb/feature_bank.base.valid.py
echo "    Models Done!"
mail -s "Train_LFB:Progress" ${USER}@sms.ed.ac.uk <<< "Synchronised Data and Models"
echo ""

# ======================
# Generate Feature Banks
# ======================


# ===========
# Train Model
# ===========
#cd ${HOME}/code/LZDetect
#echo "Training Model..."
#python -m torch.distributed.launch \
#   --nproc_per_node=${2} tools/train_net.py \
#   --config-file "${SCRATCH_HOME}/models/fcos/config.yaml" \
#   --skip-test \
#   PATHS_CATALOG "${HOME}/code/LZDetect/fcos_core/config/paths_catalog.py" \
#   MODEL.WEIGHT "${SCRATCH_HOME}/models/fcos/model.pth" \
#   DATASETS.BASE "${SCRATCH_HOME}/data" \
#   OUTPUT_DIR "${SCRATCH_HOME}/models/fcos/output" \
#   SOLVER.IMS_PER_BATCH 16 \
#   SOLVER.MAX_ITER 30000 \
#   SOLVER.CHECKPOINT_PERIOD 1000
#mail -s "Train_FCOS:Progress" ${USER}@sms.ed.ac.uk <<< "Model Training (on ${1}) Completed"

# ===========
# Copy Data
# ===========
#echo "Copying Results"
#mkdir -p "${HOME}/models/FCOS/Trained/${1}_${2}/"
#rsync --archive --update --compress "${SCRATCH_HOME}/models/fcos/output/" "${HOME}/models/FCOS/Trained/${1}_${2}/"
#mail -s "Train_FCOS:Progress" ${USER}@sms.ed.ac.uk <<< "Outputs Copied to ${1}_${2}"
#conda deactivate