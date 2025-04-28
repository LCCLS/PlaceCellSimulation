#!/bin/bash
#
# run_das6_job_CPU.sh
#
# DAS-6 SLURM submission script: CPU-only SNES, use all 48 CPUs
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=grid_cpu[%a]
#SBATCH --output=slurm-%x_%A_%a.out
#SBATCH --error=slurm-%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48       # 48 CPUs per task
#SBATCH --gres=                  # no GPUs
#SBATCH --array=0                # adjust as needed for replicates

# ───── Experiment parameters ───────────────────────────────────────────────
GRID_SIZE=${GRID_SIZE:-5}               # default grid size 5×5
REP_ID=${SLURM_ARRAY_TASK_ID}           # unique replicate ID
NUM_ACTORS=${SLURM_CPUS_PER_TASK}       # one Ray actor per CPU

echo "[⇢] Starting CPU-only task $REP_ID on $(hostname): grid=${GRID_SIZE}, actors=${NUM_ACTORS}"

# ───── Environment setup ────────────────────────────────────────────────────
. /etc/bashrc
. /etc/profile.d/lmod.sh

source /var/scratch/$USER/project/distributed_asci_supercomputer-6/venv/bin/activate

# allow each actor to use a single thread
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ───── Log hardware & config ────────────────────────────────────────────────
python - <<'PY'
import platform, datetime, os, json, sys, torch
info = {
    "host":       platform.node(),
    "cuda_available": torch.cuda.is_available(),
    "start_time": datetime.datetime.now().isoformat(timespec="seconds"),
    "python":     sys.version.split()[0],
    "num_actors": os.getenv("NUM_ACTORS"),
    "grid_size":  os.getenv("GRID_SIZE"),
    "replicate":  os.getenv("SLURM_ARRAY_TASK_ID"),
}
print(json.dumps(info, indent=2))
PY

# ───── Run the experiment ───────────────────────────────────────────────────
echo "[✓] Running SNES experiment (grid=$GRID_SIZE, rep=$REP_ID, actors=$NUM_ACTORS) on CPU..."
python -u src/main.py \
  --grid_size  "$GRID_SIZE" \
  --rep_id     "$REP_ID" \
  --num_actors "$NUM_ACTORS"

echo "[✓] Job $REP_ID finished at $(date +%F_%T)"
