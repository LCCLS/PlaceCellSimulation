#!/bin/bash

#SBATCH --job-name=grid5_test
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=24
#SBATCH --partition=defq
#SBATCH --ntasks=1

# ── Force CPU-only ──
export CUDA_VISIBLE_DEVICES=""

# ── Number of Ray actors = number of cores allocated ──
export NUM_ACTORS=$SLURM_CPUS_PER_TASK

echo "[⇢] Running single CPU-only replicate on a 5×5 grid with $NUM_ACTORS cores"

# === Configurable Arguments ===
GRID_SIZE=5
REPLICATES=1

# === RUN SCRIPT FOR DAS-6 JOB ===
echo "[✓] Job started on DAS-6..."
. /etc/bashrc
. /etc/profile.d/lmod.sh

# === Activate environment ===
echo "[✓] Activating virtual environment..."
source /var/scratch/$USER/project/distributed_asci_supercomputer-6/venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# === Print device info ===
echo "[✓] CUDA device:    $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\")')"

# === Run your experiment ===
echo "[✓] Running experiment with grid size $GRID_SIZE and $REPLICATES replicates..."
python main.py --grid_size $GRID_SIZE --replicates $REPLICATES  --num_actors $NUM_ACTORS
echo "[✓] Job completed."
