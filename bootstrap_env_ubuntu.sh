#! /bin/bash
set -euo pipefail

# ============================================================
# latent_sope bootstrap (Strategy A)
# Keep modern JAX + install robomimic, then override old pinned:
#   - diffusers==0.11.1 (too old; breaks with modern JAX)
#   - transformers==4.41.2, huggingface_hub==0.23.4
#
# One-liner:
# conda deactivate && conda remove -n latent_sope --all -y \
#   && conda create -n latent_sope python=3.10 -y \
#   && conda activate latent_sope \
#   && cd <repo_root> \
#   && bash bootstrap_env_ubuntu.sh
#
# Quick checks after install:
#   python -c "import jax; print(jax.__version__, jax.default_backend(), jax.devices())"
#   python -c "import robomimic; import robomimic.utils.file_utils as FileUtils; print('robomimic OK')"
#   python -c "import transformers, diffusers; print(transformers.__version__, diffusers.__version__)"
# ============================================================

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "[ERROR] No active conda env. Activate latent_sope before running."
  exit 1
fi

echo "[INFO] Using conda env: ${CONDA_PREFIX}"
python -V

REPO_ROOT="$(pwd)"
ROBOMIMIC_DIR="${REPO_ROOT}/third_party/robomimic"

if [[ ! -d "${ROBOMIMIC_DIR}" ]]; then
  echo "[ERROR] Expected robomimic at: ${ROBOMIMIC_DIR}"
  echo "        Check that the submodule exists and you ran from repo_root."
  exit 1
fi

# ----------------------------
# Ubuntu: ensure conda env tools take precedence
# ----------------------------
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# Avoid picking up user-local cmake shim from ~/.local/bin
if command -v cmake >/dev/null 2>&1; then
  CMAKE_BIN="$(command -v cmake)"
  if [[ "${CMAKE_BIN}" == "${HOME}/.local/bin/cmake" ]]; then
    echo "[WARN] Found ~/.local/bin/cmake shim; disabling user-local bin for this script."
    export PATH="${CONDA_PREFIX}/bin:/usr/local/bin:/usr/bin:/bin"
  fi
fi

# ----------------------------
# CMake (required by egl_probe build)
# Prefer conda-forge; fall back to pip if needed.
# ----------------------------
if ! command -v cmake >/dev/null 2>&1; then
  echo "[INFO] Installing cmake via conda-forge..."
  if command -v conda >/dev/null 2>&1; then
    conda install -y -c conda-forge cmake
  else
    echo "[WARN] conda not found; installing cmake via pip..."
    python -m pip install --upgrade cmake
  fi
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "[ERROR] cmake not found after install attempt."
  exit 1
fi

# ----------------------------
# Core tooling
# ----------------------------
python -m pip install --upgrade pip setuptools wheel

echo "[INFO] Installing egl_probe (required by robomimic)..."
python -m pip install --upgrade egl_probe

# ----------------------------
# PyTorch (CUDA 12.6)
# ----------------------------
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126

# ----------------------------
# JAX (CUDA 12) - modern
# Note: -f is the JAX CUDA wheel index
# ----------------------------
python -m pip install --upgrade \
  "jax[cuda12]" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ----------------------------
# JAX research stack + utilities
# ----------------------------
python -m pip install --upgrade \
  flax optax orbax-checkpoint \
  hydra-core einops \
  diffrax \
  pandas numpy scipy \
  matplotlib seaborn \
  tqdm rich ipython ipykernel jupyter ipywidgets \
  h5py pyyaml \
  wandb

# ----------------------------
# Mujoco / robosuite
# ----------------------------
python -m pip install --upgrade mujoco robosuite

# ----------------------------
# robomimic (editable)
# Install as usual, THEN override old huggingface/diffusers pins afterwards.
# ----------------------------
echo "[INFO] Installing robomimic from submodule (editable)..."
cd "${ROBOMIMIC_DIR}"
python -m pip install -e .
cd "${REPO_ROOT}"

# Helpful extras around datasets / video / image IO
python -m pip install --upgrade \
  opencv-python pillow imageio imageio-ffmpeg \
  tensorboard tensorboardX \
  mediapy

# ----------------------------
# Override robomimic's pinned huggingface/diffusers stack
# (This is the key fix for the KeyArray error.)
# ----------------------------
python -m pip install --upgrade \
  "diffusers>=0.28.0" \
  "transformers>=4.56.0" \
  "huggingface_hub" \
  accelerate safetensors

# ----------------------------
# Smoke tests
# ----------------------------
echo "[INFO] Verifying JAX..."
python - <<'PY'
import jax
print("JAX version:", jax.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
PY

echo "[INFO] Verifying robomimic import path you mentioned..."
python - <<'PY'
import robomimic
import robomimic.utils.file_utils as FileUtils
print("robomimic + FileUtils import OK")
PY

echo "[INFO] Verifying Hugging Face (HF) stack..."
python - <<'PY'
import transformers, diffusers
print("transformers:", transformers.__version__)
print("diffusers:", diffusers.__version__)
PY

echo "[DONE] Latent SOPE bootstrap complete."
echo "If DINOv3 weights require HF auth, run: huggingface-cli login"
