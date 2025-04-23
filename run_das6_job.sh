#!/bin/bash

#SBATCH --job-name=grid5_test
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=defq
#SBATCH -C A4000  # Request a specific GPU type if needed (e.g., A4000, A5000)

# === Configurable Arguments ===
GRID_SIZE=3
REPLICATES=10

# === RUN SCRIPT FOR DAS-6 JOB ===

echo "[✓] Job started on DAS-6..."
. /etc/bashrc
. /etc/profile.d/lmod.sh

# === Activate environment ===
echo "[✓] Activating virtual environment..."
source /var/scratch/$USER/project/distributed_asci_supercomputer-6/venv/bin/activate

# === Print device info ===
echo "[✓] CUDA device:    $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\")')"

# === Run your experiment ===
echo "[✓] Running experiment with grid size $GRID_SIZE and $REPLICATES replicates..."
python main.py --grid_size $GRID_SIZE --replicates $REPLICATES

echo "[✓] Job completed."
