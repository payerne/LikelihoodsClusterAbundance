#!/usr/bin/bash

# SLURM options:

#SBATCH --job-name=cm    # Job name
#SBATCH --output=log.log
#SBATCH --partition=htc               # Partition choice
#SBATCH --ntasks=5                    # Run a single task (by default tasks == CPU)
#SBATCH --mem=7000                    # Memory in MB per default
#SBATCH --time=0-9:00:00             # 7 days by default on htc partition
#SBATCH --array=0-7

ID=$( $SLURM_ARRAY_TASK_ID )
INDEX=$(($SLURM_ARRAY_TASK_ID))

source /pbs/home/c/cpayerne/setup_mydesc.sh

python compute_likelihood_cosmo_samples_mcmc.py  $INDEX 

