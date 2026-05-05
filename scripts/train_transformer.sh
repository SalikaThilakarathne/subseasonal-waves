#!/bin/bash

#################################################################
# batch settings for job on levante at DKRZ, metascales model run
# Mar, 2026
#################################################################
#SBATCH --job-name=pytorch-wave         # Job name
#SBATCH --partition=gpu       # Partition name
#SBATCH --mem=100G                      # Memory per node
#SBATCH --gres=gpu:a100_40
#SBATCH --time=12:00:00               # Time limit (hh:mm:ss) #if we want more, make a request to support@dkrz.de
#SBATCH --output=log/job_%j.out           # Standard output log
#SBATCH --error=log/job_%j.err            # Standard error log
#SBATCH --account=gg0303     # Account name

module load pytorch

# go up one directory, then into 'codes'
cd ../codes

# run the python grid-search script
python -u slurm_train_job.py


