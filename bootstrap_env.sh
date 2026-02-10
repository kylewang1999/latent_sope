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
#   && bash bootstrap.sh
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
CLEAN_DIFFUSER_DIR="${REPO_ROOT}/third_party/clean_diffuser"

if [[ ! -d "${ROBOMIMIC_DIR}" ]]; then
  echo "[ERROR] Expected robomimic at: ${ROBOMIMIC_DIR}"
  echo "        Check that the submodule exists and you ran from repo_root."
  exit 1
fi

if [[ ! -d "${CLEAN_DIFFUSER_DIR}" ]]; then
  echo "[ERROR] Expected clean_diffuser at: ${CLEAN_DIFFUSER_DIR}"
  echo "        Check that the submodule exists and you ran from repo_root."
  exit 1
fi

# ----------------------------
# CARC: load cmake via modules
# ----------------------------
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
  fi
fi

if command -v module >/dev/null 2>&1; then
  echo "[INFO] Loading cmake module..."
  module load cmake
else
  echo "[WARN] module command not found; assuming cmake already on PATH."
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "[ERROR] cmake not found. On CARC you likely need: module load cmake"
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
# Mujoco / robosuite / gym
# ----------------------------
python -m pip install --upgrade mujoco robosuite pygame

# ----------------------------
# MuJoCo 2.1 + mujoco-py (required by d4rl)
# ----------------------------
MUJOCO_DIR="${HOME}/.mujoco/mujoco210"
MUJOCO_TAR="${MUJOCO_DIR}.tar.gz"
MUJOCO_URL="https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_DIR}"
export LD_LIBRARY_PATH="${MUJOCO_DIR}/bin:${LD_LIBRARY_PATH:-}"

if [[ ! -d "${MUJOCO_DIR}" ]]; then
  echo "[INFO] MuJoCo 2.1 not found. Downloading..."
  mkdir -p "${HOME}/.mujoco"
  if command -v curl >/dev/null 2>&1; then
    curl -L "${MUJOCO_URL}" -o "${MUJOCO_TAR}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${MUJOCO_TAR}" "${MUJOCO_URL}"
  else
    echo "[ERROR] Neither curl nor wget is available to download MuJoCo."
    exit 1
  fi
  tar -xzf "${MUJOCO_TAR}" -C "${HOME}/.mujoco"
  rm -f "${MUJOCO_TAR}"
fi

if ! grep -Fxq 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kyle/.mujoco/mujoco210/bin' "$HOME/.bashrc"; then
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kyle/.mujoco/mujoco210/bin' >> "$HOME/.bashrc"
fi

python -m pip install -U 'mujoco-py<2.2,>=2.1'

# ----------------------------
# robomimic (editable)
# Install as usual, THEN override old huggingface/diffusers pins afterwards.
# ----------------------------
echo "[INFO] Installing robomimic from submodule (editable)..."
cd "${ROBOMIMIC_DIR}"
python -m pip install -e .
cd "${REPO_ROOT}"

echo "[INFO] Installing clean_diffuser from submodule (editable)..."
cd "${CLEAN_DIFFUSER_DIR}"
# python -m pip install -e .
python -m pip install -e . --no-deps --no-build-isolation # ignore dependsncies
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

# For stitch-ope repo
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
