#!/bin/bash
# Submit a notebook to run on the GPU cluster.
#
# Usage:
#   bash scripts/submit_notebook.sh experiments/2026-03-10/500k_steps.ipynb
#
# Check status:
#   squeue -u $USER
#
# View output:
#   cat slurm_logs/<job_id>_500k_steps.out
#
# Cancel:
#   scancel <job_id>

set -e

NOTEBOOK="${1:?Usage: bash scripts/submit_notebook.sh <notebook.ipynb>}"

if [ ! -f "$NOTEBOOK" ]; then
    echo "ERROR: Notebook not found: $NOTEBOOK"
    exit 1
fi

cd ~/latent_sope
mkdir -p slurm_logs

JOB_NAME=$(basename "$NOTEBOOK" .ipynb)

echo "Submitting: $NOTEBOOK"
echo "Job name:   $JOB_NAME"

sbatch --job-name="$JOB_NAME" scripts/run_notebook.sbatch "$NOTEBOOK"

echo ""
echo "Check status:  squeue -u \$USER"
echo "View output:   ls -t slurm_logs/ | head -2"
echo "Cancel:        scancel <job_id>"
