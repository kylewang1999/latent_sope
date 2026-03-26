#!/bin/bash
#SBATCH --job-name=500k_steps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

# ── Usage ──────────────────────────────────────────────────
#   cd ~/latent_sope
#   bash scripts/submit_notebook.sh experiments/2026-03-10/500k_steps.ipynb
#
# Or directly:
#   sbatch scripts/run_notebook.sbatch experiments/2026-03-10/500k_steps.ipynb
# ───────────────────────────────────────────────────────────

set -e

NOTEBOOK="${1:?Usage: sbatch run_notebook.sbatch <notebook.ipynb>}"

# Verify notebook exists
if [ ! -f "$NOTEBOOK" ]; then
    echo "ERROR: Notebook not found: $NOTEBOOK"
    exit 1
fi

echo "=== SLURM Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Notebook:  $NOTEBOOK"
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

# Remove ~/.local/bin from PATH to avoid picking up wrong jupyter
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "$HOME/.local/bin" | tr '\n' ':' | sed 's/:$//')

# Verify we're using the conda jupyter
echo "Using jupyter: $(which jupyter)"
echo "Python: $(which python)"

# Run notebook in-place (saves outputs into the .ipynb)
cd ~/latent_sope
jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    "$NOTEBOOK" \
    --ExecutePreprocessor.timeout=-1

echo ""
echo "=== DONE: $(date) ==="
