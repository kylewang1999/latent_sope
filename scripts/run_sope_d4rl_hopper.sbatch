#!/bin/bash
#SBATCH --job-name=sope_d4rl_hopper
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -e

echo "=== SLURM Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Started:   $(date)"
echo "======================"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate latent_sope

# Set required env vars
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "$HOME/.local/bin" | tr '\n' ':' | sed 's/:$//')

echo "Python: $(which python)"

cd ~/latent_sope/third_party/sope

python opelab/examples/d4rl/main_full.py \
    --config opelab/examples/d4rl/configs/hopper_stitch.json \
    --device cuda

echo ""
echo "=== DONE: $(date) ==="
