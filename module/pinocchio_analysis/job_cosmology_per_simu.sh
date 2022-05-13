#!/usr/bin/bash

# SLURM options:

# SBATCH --job-name=test    # Job name
# SBATCH --output=log.log
# SBATCH --partition=htc               # Partition choice
# SBATCH --ntasks=1                    # Run a single task (by default tasks == CPU)
# SBATCH --mem=7000                    # Memory in MB per default
# SBATCH --time=1-07:00:00             # 7 days by default on htc partition
# SBATCH --array=1-500

N_JOB=$SLURM_ARRAY_TASK_MAX
N_SIMU=500
N_PER_JOB=$(($N_SIMU / $N_JOB))
N_Z_BIN=100
N_M_BIN=100
START=500
ID=$(( $START+$SLURM_ARRAY_TASK_ID ))
echo startof $ID
# Print the task and run range

source /pbs/home/c/cpayerne/setup_mydesc.sh
python /pbs/throng/lsst/users/cpayerne/ClusterLikelihoods/modules/pinocchio_analysis/compute_posteriors_per_simu.py $N_Z_BIN $N_M_BIN MPG $ID


