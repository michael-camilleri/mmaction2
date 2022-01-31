#!/bin/bash
#  Author: Michael Camilleri
#
#  Scope:
#     Fine-Tunes the LFB model (no explicit LFB/Backbone training beyond what goes on in backprop)
#
#  Script takes two parameters
#     A/C - Choice between running on Apollo or Charles Nodes
#     [Cores] - Number of GPU Threads to use
#
#  USAGE:
#     bash train_lfb.sh
#
#  Data Structures
#     Data is expected to be under ${HOME}/data/LFB/[DATASET] where [DATASET] is Train/Validate
#     Model PTHs/Configs are under ${HOME}/models/LFB/Base/[models/configs] respectively


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

# ================================
# Download Data and Models if necessary
# ================================
echo "Consolidating Data/Models in ${SCRATCH_HOME}"
echo "  -> Synchronising Data"
mkdir -p ${SCRATCH_HOME}/data/behaviour
rsync --archive --update --compress "${HOME}/data/LFB/" ${SCRATCH_HOME}/data/behaviour/
echo "    Data Done!"
echo "  -> Synchronising Model"
mkdir -p ${SCRATCH_HOME}/models/behaviour/
rsync --archive --update --compress ${HOME}/models/LFB/Base/ ${SCRATCH_HOME}/models/behaviour/
echo "    Models Done!"
mail -s "Train_LFB:Progress" ${USER}@sms.ed.ac.uk <<< "On ${1}: Synchronised Data (from ${2}) and
Model"

# ======================
# Generate Feature Banks
# ======================


# ===========
# Train Model
# ===========
cd ${HOME}/code/LZDetect
echo "Training Model..."
python -m torch.distributed.launch \
   --nproc_per_node=${3} tools/train_net.py \
   --config-file "${SCRATCH_HOME}/models/fcos/config.yaml" \
   --skip-test \
   PATHS_CATALOG "${HOME}/code/LZDetect/fcos_core/config/paths_catalog.py" \
   MODEL.WEIGHT "${SCRATCH_HOME}/models/fcos/model.pth" \
   DATASETS.BASE "${SCRATCH_HOME}/data" \
   OUTPUT_DIR "${SCRATCH_HOME}/models/fcos/output" \
   SOLVER.IMS_PER_BATCH 16 \
   SOLVER.MAX_ITER 30000 \
   SOLVER.CHECKPOINT_PERIOD 1000
mail -s "Train_FCOS:Progress" ${USER}@sms.ed.ac.uk <<< "Model Training (on ${1}) Completed"

# ===========
# Copy Data
# ===========
echo "Copying Results"
mkdir -p "${HOME}/models/FCOS/Trained/${1}_${2}/"
rsync --archive --update --compress "${SCRATCH_HOME}/models/fcos/output/" "${HOME}/models/FCOS/Trained/${1}_${2}/"
mail -s "Train_FCOS:Progress" ${USER}@sms.ed.ac.uk <<< "Outputs Copied to ${1}_${2}"
conda deactivate