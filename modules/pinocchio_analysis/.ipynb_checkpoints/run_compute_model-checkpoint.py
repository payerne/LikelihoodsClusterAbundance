import numpy as np
import os
for i in range(100):
    down, up=int(i*100), int((i+1)*100-1)
    print(down, up)
    os.system(f'python /pbs/throng/lsst/users/cpayerne/ClusterLikelihoods/modules/pinocchio_analysis/compute_model.py 100 100 {down} {up}')
    print('done')
   # $ compute_model.py 100 100 $down $up
