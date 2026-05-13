#!/bin/bash
#SBATCH -p compute                # Queue/Partition (check Lanta docs if 'compute' is wrong)
#SBATCH -N 1                      # 1 Node
#SBATCH -c 8                      # 8 CPU Cores
#SBATCH --gres=gpu:1              # Request 1 GPU (e.g., A100)
#SBATCH -t 48:00:00               # Max time allowed (48 hours)
#SBATCH -A lt200321               # Lanta Project ID/Account
#SBATCH -J aasist_mms             # Name of the job
#SBATCH -o training_log.out       # Where to save the normal print outputs
#SBATCH -e training_error.err     # Where to save error messages

# Load modules and activate environment
module purge
module load cray-python/3.11.7
source /lustrefs/disk/home/pdenninn/nectecsimpletest/.venv/bin/activate

# Navigate to code folder
cd /lustrefs/disk/home/pdenninn/nectecsimpletest

# Run the training script!
python main.py --config config/S4_aasist3_full.conf
