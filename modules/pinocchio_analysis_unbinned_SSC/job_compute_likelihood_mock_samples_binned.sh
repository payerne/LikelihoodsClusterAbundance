#!/usr/bin/bash

# SLURM options:

#SBATCH --job-name=cm    # Job name
#SBATCH --output=log.log
#SBATCH --partition=htc               # Partition choice
#SBATCH --ntasks=5                    # Run a single task (by default tasks == CPU)
#SBATCH --mem=7000                    # Memory in MB per default
#SBATCH --time=0-1:00:00             # 7 days by default on htc partition
#SBATCH --array=0-18

ID=$( $SLURM_ARRAY_TASK_ID )
INDEX=$(($SLURM_ARRAY_TASK_ID))

source /pbs/home/c/cpayerne/setup_mydesc.sh
python /pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/pinocchio_analysis_unbinned_SSC/compute_likelihood_binned_mock_samples.py binned_gaussian  $INDEX 


