#!/usr/bin/bash

# SLURM options:

#SBATCH --job-name=cm    # Job name
#SBATCH --output=log.log
#SBATCH --partition=htc               # Partition choice
#SBATCH --ntasks=5                    # Run a single task (by default tasks == CPU)
#SBATCH --mem=7000                    # Memory in MB per default
#SBATCH --time=0-1:00:00             # 7 days by default on htc partition
#SBATCH --array=1-4

ID=$( $SLURM_ARRAY_TASK_ID )
INDEX=$(($SLURM_ARRAY_TASK_ID - 1))

source /pbs/home/c/cpayerne/setup_mydesc.sh
python /pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC/SSC_contribution/SSC_contribution_hybrid.py $INDEX

