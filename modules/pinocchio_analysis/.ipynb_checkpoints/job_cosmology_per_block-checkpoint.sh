#!/usr/bin/bash

# !/bin/sh

# SLURM options:

#SBATCH --job-name=post    # Job name
#SBATCH --output=log_cosmology.log
#SBATCH --partition=htc               # Partition choice
#SBATCH --ntasks=1                    # Run a single task (by default tasks == CPU)
#SBATCH --mem=8G                    # Memory in MB per default
#SBATCH --time=0-7:00:00             # 7 days by default on htc partition
#SBATCH --array=1-50
#SBATCH --mail-user=constantin.payerne@lpsc.in2p3.fr
#SBATCH --mail-type=END,FAIL 

N_JOB=$SLURM_ARRAY_TASK_MAX
N_SIMU=1000
N_PER_JOB=$(($N_SIMU / $N_JOB))
N_Z_BIN=100
N_M_BIN=100

START_NUM=$(( ($SLURM_ARRAY_TASK_ID-1) * $N_PER_JOB ))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $N_PER_JOB - 1))

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID, which will do runs from $START_NUM to $END_NUM

source /pbs/home/c/cpayerne/setup_mydesc.sh
python /pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/pinocchio_analysis/compute_posteriors_per_block.py $N_Z_BIN $N_M_BIN GaussianCholesky $START_NUM $END_NUM 


