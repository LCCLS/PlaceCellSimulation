#!/bin/bash

# === Configurable arguments ===
GRID_SIZE=3
REPLICATES=2

# Navigate to the script's directory
cd "$(dirname "$0")"

echo "[✓] Inside project folder: $PWD"

# Create virtual environment if missing
if [ ! -d "venv" ]; then
    echo "[✓] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
echo "[✓] Activating virtual environment..."
source venv/bin/activate

# Install core packages
echo "[✓] Installing base packages..."
pip install --upgrade pip
pip install -r requirements.txt
pip install evotorch

# Check if evotorch is installed, if not, install it
if ! python3 -c "import evotorch" &> /dev/null; then
    echo "[✓] Installing evotorch..."
    pip install evotorch
else
    echo "[✓] evotorch already installed."
fi

# Run your experiment
echo "[✓] Running experiment with grid size $GRID_SIZE and $REPLICATES replicates..."
python3 main.py --grid_size $GRID_SIZE --replicates $REPLICATES

echo "[✓] All done!"
