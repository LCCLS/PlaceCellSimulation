#!/bin/bash
#
# run_das6_job_GPU.sh
#
# DAS-6 SLURM submission script: SNES on GPU‐only
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=grid_gpu[%a]
#SBATCH --output=slurm-%x_%A_%a.out
#SBATCH --error=slurm-%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4         # minimal CPU for driver threads
#SBATCH --gres=gpu:1              # allocate 1 GPU
#SBATCH -C A4000
#SBATCH --array=0-9

# ───── Experiment parameters ───────────────────────────────────────────────
GRID_SIZE=${GRID_SIZE:-8}               # default grid 5×5
REP_ID=${SLURM_ARRAY_TASK_ID}           # replicate ID
NUM_ACTORS=0                            # 0 ⇒ GPU‐only evaluation (SNES)

echo "[⇢] Starting GPU‐only task $REP_ID on $(hostname): grid=${GRID_SIZE}, actors=${NUM_ACTORS}"

# ───── Environment setup ────────────────────────────────────────────────────
. /etc/bashrc
. /etc/profile.d/lmod.sh
module load cuda12.3/toolkit            # match CUDA version of your venv
source /var/scratch/$USER/project/distributed_asci_supercomputer-6/venv/bin/activate

# prevent nested threading in PyTorch / MKL
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ───── Log hardware & config ────────────────────────────────────────────────
python - <<'PY'
import torch, platform, datetime, os, json, sys
info = {
    "host":           platform.node(),
    "cuda_available": torch.cuda.is_available(),
    "device":         torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "start_time":     datetime.datetime.now().isoformat(timespec="seconds"),
    "python_version": sys.version.split()[0],
    "num_actors":     os.getenv("NUM_ACTORS"),
    "grid_size":      os.getenv("GRID_SIZE"),
    "replicate":      os.getenv("SLURM_ARRAY_TASK_ID"),
}
print(json.dumps(info, indent=2))
PY

# ───── Run the experiment ───────────────────────────────────────────────────
echo "[✓] Running SNES experiment (grid=$GRID_SIZE, rep=$REP_ID) on GPU‐only..."
python -u src/main.py \
  --grid_size  "$GRID_SIZE" \
  --rep_id     "$REP_ID" \
  --num_actors "$NUM_ACTORS"

echo "[✓] Job $REP_ID finished at $(date +%F_%T)"