#!/bin/bash
# Pipeline: download raw mh demos → render image datasets → train diffusion policy
# Tasks: lift/mh and can/mh
# Config: prediction_horizon=16, action_horizon=1, 7 checkpoints at epochs 50,100,200,300,400,500,600
set -euo pipefail

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
export LD_LIBRARY_PATH="/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

REPO_ROOT="/workspace/latent_sope"
ROBOMIMIC_DIR="${REPO_ROOT}/third_party/robomimic"
DATASET_DIR="${ROBOMIMIC_DIR}/datasets"
SCRIPTS_DIR="${ROBOMIMIC_DIR}/robomimic/scripts"
CONFIGS_DIR="${REPO_ROOT}/configs/robomimic"

mkdir -p "${DATASET_DIR}/lift/mh"
mkdir -p "${DATASET_DIR}/can/mh"

# ─── Step 1: Download raw demos ────────────────────────────────────────────────
echo "========================================================"
echo "STEP 1: Downloading raw MH demo files"
echo "========================================================"

cd "${ROBOMIMIC_DIR}"

python -m robomimic.scripts.download_datasets \
    --tasks lift can \
    --dataset_types mh \
    --hdf5_types raw \
    --download_dir "${DATASET_DIR}"

# ─── Step 2: Render image observations ─────────────────────────────────────────
echo ""
echo "========================================================"
echo "STEP 2: Rendering image observations (84x84, agentview + wrist)"
echo "========================================================"

cd "${SCRIPTS_DIR}"

for TASK in lift can; do
    RAW="${DATASET_DIR}/${TASK}/mh/demo_v15.hdf5"
    OUT="${DATASET_DIR}/${TASK}/mh/image_v15.hdf5"

    if [ -f "${OUT}" ]; then
        echo "[${TASK}] image_v15.hdf5 already exists — skipping render"
    else
        echo "[${TASK}] Rendering images from ${RAW} ..."
        python dataset_states_to_obs.py \
            --done_mode 2 \
            --dataset "${RAW}" \
            --output_name image_v15.hdf5 \
            --camera_names agentview robot0_eye_in_hand \
            --camera_height 84 \
            --camera_width 84
        echo "[${TASK}] Done rendering → ${OUT}"
    fi
done

# ─── Step 3: Train diffusion policy for each task ──────────────────────────────
echo ""
echo "========================================================"
echo "STEP 3: Training diffusion policy"
echo "  prediction_horizon=16, action_horizon=1"
echo "  Checkpoints at epochs: 50, 100, 200, 300, 400, 500, 600"
echo "========================================================"

cd "${ROBOMIMIC_DIR}"

for TASK in lift can; do
    CONFIG="${CONFIGS_DIR}/diffusion_policy_${TASK}_mh_image.json"
    echo ""
    echo "--- Training on ${TASK}/mh ---"
    python -m robomimic.scripts.train --config "${CONFIG}"
    echo "--- Done training ${TASK} ---"
done

echo ""
echo "========================================================"
echo "ALL DONE. Checkpoints saved under:"
echo "  ${ROBOMIMIC_DIR}/diffusion_policy_trained_models/lift_mh/"
echo "  ${ROBOMIMIC_DIR}/diffusion_policy_trained_models/can_mh/"
echo "========================================================"
