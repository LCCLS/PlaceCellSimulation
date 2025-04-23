#!/bin/bash

# === SETUP SCRIPT FOR DAS-6 ENVIRONMENT ===
# Run this manually on fs0.das6.cs.vu.nl (not via sbatch)

echo "[✓] Setting up DAS-6 Python environment..."

# === Load base environment ===
. /etc/bashrc
. /etc/profile.d/lmod.sh
module load cuda12.1/toolkit  # or whatever CUDA version DAS-6 supports

# === Configuration ===
PYTHON_VERSION=3.10.13
HOME_INSTALL_DIR=$HOME/local
PYTHON_PREFIX=$HOME_INSTALL_DIR/python
SCRATCH_PROJECT_DIR=/var/scratch/$USER/project/distributed_asci_supercomputer-6
VENV_DIR=$SCRATCH_PROJECT_DIR/venv
REQUIREMENTS_FILE=$SCRATCH_PROJECT_DIR/requirements.txt

# === Install Python to $HOME/local if not already present ===
if [ ! -x "$PYTHON_PREFIX/bin/python3" ]; then
    echo "[✓] Python $PYTHON_VERSION not found — installing to $PYTHON_PREFIX..."
    mkdir -p $HOME_INSTALL_DIR
    cd $HOME_INSTALL_DIR

    wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz
    tar -xzf Python-$PYTHON_VERSION.tgz
    cd Python-$PYTHON_VERSION

    ./configure --prefix=$PYTHON_PREFIX --enable-optimizations --with-ensurepip=install
    make -j $(nproc)
    make install
else
    echo "[✓] Python already installed at $PYTHON_PREFIX"
fi

# === Create venv in SCRATCH ===
if [ -d "$VENV_DIR" ]; then
    echo "[!] Virtualenv already exists at $VENV_DIR. Skipping creation."
else
    echo "[✓] Creating virtual environment at: $VENV_DIR"
    $PYTHON_PREFIX/bin/python3 -m venv $VENV_DIR
fi

# === Activate venv and install requirements ===
echo "[✓] Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "[✓] Upgrading pip..."
pip install --upgrade pip

echo "[✓] Installing dependencies from $REQUIREMENTS_FILE..."
pip install -r $REQUIREMENTS_FILE

echo "[✓] Setup complete!"
echo "    Python:      $PYTHON_PREFIX"
echo "    Virtualenv:  $VENV_DIR"

python -c "
import torch
print('[✓] torch version:', torch.__version__)
if torch.cuda.is_available():
    print('[✓] CUDA is available! Using:', torch.cuda.get_device_name(0))
else:
    print('[!] CUDA is NOT available. Check driver/module settings.')
"