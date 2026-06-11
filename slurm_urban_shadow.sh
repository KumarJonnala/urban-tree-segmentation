#!/bin/bash
#
# Examples:
#   sbatch slurm_urban_shadow.sh
#   sbatch --export=COMMAND=all,TILE_SIZE=all                                        slurm_urban_shadow.sh
#   sbatch --export=COMMAND=shadow,TILE_SIZE=250,DATETIME="2026-06-21T10:00:00"      slurm_urban_shadow.sh
#   sbatch --export=COMMAND=shadow,TILE_SIZE=all,DATETIME="2026-06-21T12:00:00"      slurm_urban_shadow.sh
#   sbatch --export=COMMAND=segment,TILE_SIZE=100                                    slurm_urban_shadow.sh
#   sbatch --export=COMMAND=merge,TILE_SIZE=250                                      slurm_urban_shadow.sh
#   sbatch --export=COMMAND=render,TILE_SIZE=250                                     slurm_urban_shadow.sh
#   sbatch --export=COMMAND=status                                                   slurm_urban_shadow.sh
#

#SBATCH --job-name=urban_shadow
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

#SBATCH --partition=gpu
#SBATCH --nodelist=ant1
#SBATCH --gres=shard:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

# --- environment ---
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

pip install -q -r requirements.txt

# --- pipeline ---
# Edit TILE_SIZE and COMMAND below, or pass as sbatch --export= variables
TILE_SIZE=${TILE_SIZE:-250}
COMMAND=${COMMAND:-all}
DATETIME=${DATETIME:-}          # e.g. "2026-06-21T10:00:00"  (UTC); empty = now

echo "Starting urban-shadow-analysis pipeline"
echo "  Command   : $COMMAND"
echo "  Tile size : ${TILE_SIZE}"
echo "  Datetime  : ${DATETIME:-now (UTC)}"
echo "  Node      : $SLURMD_NODENAME"
echo "  Job ID    : $SLURM_JOB_ID"
echo "  Time      : $(date -u '+%Y-%m-%d %H:%M UTC')"

# Build argument list
ARGS=()
if [ "$TILE_SIZE" = "all" ]; then
    ARGS+=(--all-sizes)
else
    ARGS+=(--tile-size "$TILE_SIZE")
fi
if [ -n "$DATETIME" ]; then
    ARGS+=(--datetime-utc "$DATETIME")
fi

python3 pipeline.py "$COMMAND" "${ARGS[@]}"

echo "Done: $(date -u '+%Y-%m-%d %H:%M UTC')"
