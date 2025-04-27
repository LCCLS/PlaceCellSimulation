#!/bin/bash

########################################
# 🔐 1. LOGIN TO DAS-6
########################################

# From your machine (Berlin etc.)
ssh -J lcn247@ssh.data.vu.nl lcn247@fs0.das6.cs.vu.nl

########################################
# 📦 2. UPLOAD EXPERIMENT
########################################

# Run locally (Mac/your system)
tar -czf distributed_asci_supercomputer-6.tar.gz distributed_asci_supercomputer-6/
scp -o ProxyJump=lcn247@ssh.data.vu.nl distributed_asci_supercomputer-6.tar.gz lcn247@fs0.das6.cs.vu.nl:~

#scratch dir 
scp -o ProxyJump=lcn247@ssh.data.vu.nl distributed_asci_supercomputer-6.tar.gz lcn247@fs0.das6.cs.vu.nl:/var/scratch/lcn247/project/


# Then on DAS-6 (fs0)
tar -xzf distributed_asci_supercomputer-6.tar.gz
cd distributed_asci_supercomputer-6/

########################################
# ⚙️ 3. SET UP PYTHON VENV (Run Once)
########################################

bash setup_das6_env.sh

########################################
# 🚀 4. RUN JOB
########################################

# Run SLURM job with optional env vars
GRID_SIZE=5 REPLICATES=10 sbatch run_das6_job.sh

########################################
# 📊 5. MONITORING & DEBUGGING
########################################

# Check current job(s)
squeue -u lcn247

# Tail job logs
tail -f slurm-1164354.out
tail -f slurm-<jobid>.err

scontrol show job 1164354
# Cancel job
scancel <1164354>

# View past jobs
sacct -u lcn247 --format=JobID,JobName,State,Elapsed,MaxRSS,AllocGRES

########################################
# 📥 6. DOWNLOAD RESULTS TO LOCAL
########################################

scp -r -o ProxyJump=lcn247@ssh.data.vu.nl lcn247@fs0.das6.cs.vu.nl:/var/scratch/lcn247/project/distributed_asci_supercomputer-6/grid_5 ./

########################################
# 🔍 7. CHECK AVAILABLE RESOURCES
########################################

# List partitions and node status
sinfo

# filter on GPU
sinfo -o "%20N %10P %10G %20f" | grep gpu

# Detailed view of GPU availability
scontrol show nodes | grep -e NodeName -e Gres -e State

# View nodes by partition and GPU model
sinfo -o "%20N %10P %5c %10m %10G %20f %E"

########################################
# ✅ 8. CHECK PYTHON / CUDA SETUP
########################################

# Check Python version inside venv
source venv/bin/activate
python --version

# Check GPU availability (inside Python script or interactive shell)
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

########################################
# 🧹 9. CLEANUP
########################################

# On DAS-6
rm -rf distributed_asci_supercomputer-6.tar.gz
rm -rf grid_*
rm -f slurm-*.out slurm-*.err

# remove files your machinels
rm distributed_asci_supercomputer-6.tar.gz


#check directory 
pwd


#move folder over to scratch directory 
mv ~/distributed_asci_supercomputer-6 /var/scratch/$USER/project/



# exectuable scripts 
chmod +x run_das6_job.sh
chmod +x setup_das6_env.sh


  