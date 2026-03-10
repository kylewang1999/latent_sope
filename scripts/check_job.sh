#!/bin/bash
# Quick status check for SLURM jobs.
#
# Usage:
#   bash scripts/check_job.sh          # check all your jobs
#   bash scripts/check_job.sh 7267244  # check specific job + tail log

echo "=== Your SLURM Jobs ==="
JOBS=$(squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %.6D %R" 2>/dev/null)
if [ "$(echo "$JOBS" | wc -l)" -le 1 ]; then
    echo "No running jobs."
else
    echo "$JOBS"
fi
echo ""

# If a job ID was given, show its log
if [ -n "$1" ]; then
    JOB_ID="$1"
    OUT=$(ls -t slurm_logs/${JOB_ID}_*.out 2>/dev/null | head -1)
    ERR=$(ls -t slurm_logs/${JOB_ID}_*.err 2>/dev/null | head -1)

    if [ -n "$ERR" ] && [ -s "$ERR" ]; then
        echo "=== ERRORS ($ERR) ==="
        tail -20 "$ERR"
        echo ""
    fi

    if [ -n "$OUT" ]; then
        echo "=== Last 30 lines of output ($OUT) ==="
        tail -30 "$OUT"
    else
        echo "No log file found for job $JOB_ID"
    fi
else
    # Show the most recent log
    LATEST=$(ls -t slurm_logs/*.out 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "=== Latest log: $LATEST ==="
        tail -20 "$LATEST"
    fi
fi
