conda activate latent_sope

# install MuJoCo 2.1 if missing
MUJOCO_DIR="$HOME/.mujoco/mujoco210"
MUJOCO_TAR="${MUJOCO_DIR}.tar.gz"
MUJOCO_URL="https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"

if [[ ! -d "${MUJOCO_DIR}" ]]; then
  mkdir -p "$HOME/.mujoco"
  curl -L "$MUJOCO_URL" -o "$MUJOCO_TAR"
  tar -xzf "$MUJOCO_TAR" -C "$HOME/.mujoco"
  rm -f "$MUJOCO_TAR"
fi

export MUJOCO_PY_MUJOCO_PATH="$MUJOCO_DIR"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}"

if ! grep -Fxq 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kyle/.mujoco/mujoco210/bin' "$HOME/.bashrc"; then
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kyle/.mujoco/mujoco210/bin' >> "$HOME/.bashrc"
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> "$HOME/.bashrc"
fi

python -m pip install -U 'mujoco-py<2.2,>=2.1'
python -m pip install -U 'git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl'
