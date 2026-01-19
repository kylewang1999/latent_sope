#! /bin/bash
set -euo pipefail

# ============================================================
#
# Installs conda activate/deactivate hooks so MuJoCo/robosuite
# uses headless EGL rendering by default.
#
# Usage (from anywhere, after conda activate latent_sope):
#   bash scripts/setup_mujoco_egl.sh
#
# You can override per job:
#   export MUJOCO_GL=egl
#   export MUJOCO_EGL_DEVICE_ID=0
#   export PYOPENGL_PLATFORM=egl
# ============================================================

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "[ERROR] No active conda env. Activate your env before running."
  exit 1
fi

ACTIVATE_D="${CONDA_PREFIX}/etc/conda/activate.d"
DEACTIVATE_D="${CONDA_PREFIX}/etc/conda/deactivate.d"

mkdir -p "${ACTIVATE_D}" "${DEACTIVATE_D}"

echo "[INFO] Writing conda hooks to:"
echo "  ${ACTIVATE_D}/mujoco_egl.sh"
echo "  ${DEACTIVATE_D}/mujoco_egl.sh"


# Write into activate.d/mujoco_egl.sh
cat > "${ACTIVATE_D}/mujoco_egl.sh" <<'SH'
# Prefer headless GPU rendering for MuJoCo on Linux
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Helps some OpenGL stacks (PyOpenGL) pick EGL on headless nodes
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Select GPU for EGL if multiple GPUs exist (0 by default).
# You can override in your job script: export MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

echo "[activate] MUJOCO_GL=$MUJOCO_GL PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
SH

# Write into deactivate.d/mujoco_egl.sh
cat > "${DEACTIVATE_D}/mujoco_egl.sh" <<'SH'
unset MUJOCO_GL
unset PYOPENGL_PLATFORM
unset MUJOCO_EGL_DEVICE_ID
SH

echo "[DONE] MuJoCo EGL conda hooks installed."
echo "Next: run 'conda deactivate && conda activate <env>' to apply in your shell."
