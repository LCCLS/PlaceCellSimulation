#!/bin/bash

# === Hard-coded Speedup Comparison Script ===
# Runs the experiment twice: once with 1 CPU actor, once with 8 actors,
# then prints a summary of the two runtimes.

# Configurable arguments
GRID_SIZE=${GRID_SIZE:3}
REPLICATES=${REPLICATES:1}

# Navigate to script directory
cd "$(dirname "$0")"
echo "[✓] Inside project folder: $PWD"

# Ensure and activate venv
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
pip install --upgrade pip >&2
pip install -r requirements.txt >&2
pip install evotorch >&2

# Prevent BLAS/OpenMP oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Function to run and time one experiment.
# Verbose logs go to stderr; only elapsed seconds to stdout.
run_and_time() {
    local actors=$8
    >&2 echo
    >&2 echo "[▶] Starting run with $actors actor(s)..."
    local start=$(date +%s)
    python3 main.py \
      --grid_size "$GRID_SIZE" \
      --replicates "$REPLICATES" \
      --num_actors "$actors" >&2
    local end=$(date +%s)
    local elapsed=$((end - start))
    >&2 echo "[⏱] Completed with $actors actor(s) in ${elapsed}s"
    echo "$elapsed"
}

# Run experiments with 1 and 8 actors
elapsed1=$(run_and_time 1)
elapsed2=$(run_and_time 8)

# Final summary
echo
echo "[✓] Speedup comparison done."
echo "[Summary] 1 actor run took: ${elapsed1}s"
echo "[Summary] 8 actor(s) run took: ${elapsed2}s"