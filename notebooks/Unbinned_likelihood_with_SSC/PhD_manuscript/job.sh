#!/usr/bin/bash

# SLURM options:

#SBATCH --job-name=cm    # Job name
#SBATCH --output=log.log
#SBATCH --partition=htc               # Partition choice
#SBATCH --ntasks=5                    # Run a single task (by default tasks == CPU)
#SBATCH --mem=7000                    # Memory in MB per default
#SBATCH --time=0-20:00:00             # 7 days by default on htc partition
#SBATCH --array=1-3

source /pbs/home/c/cpayerne/setup_mydesc.sh
#python /pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC/manuscript/mcmc_unbinned_SSC.py $SLURM_ARRAY_TASK_ID noSSC

python /pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC/manuscript/mcmc_unbinned_SSC.py $SLURM_ARRAY_TASK_ID


