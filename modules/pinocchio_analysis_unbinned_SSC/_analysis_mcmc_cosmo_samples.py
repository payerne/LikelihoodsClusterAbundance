import numpy as np
import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import PINOCCHIO_cat
where_samples = '/pbs/throng/lsst/users/cpayerne/ClusterLikelihoods_data/SPT_eRosita_Planck/'
analysis_name = ['spt', 'planck', 'erosita', 'pinocchio',]
analysis_name_file = [where_samples+'spt.pkl', 
                      where_samples+'planck.pkl',
                      where_samples+'erosita.pkl',
                      where_samples+'pinocchio.pkl', ]
fsky = [ (2500/(360**2/np.pi)), 0.5,1,0.25,]
n_cat = [1,2,4,1]
reshape_sky = [True, False, False, False]
likelihood_name = ['hybrid_garrell_unbinned_SSC','full_unbinned',]
RZ = []
RM = []
NAME = []
LIKELIHOOD = []
for i in range(len(likelihood_name)):
    for j in range(len(analysis_name)):
        LIKELIHOOD.append(likelihood_name[i])
        NAME.append(analysis_name[j])
analysis_unbinned = {'analysis_name':NAME, 'likelihood':LIKELIHOOD}
print(analysis_unbinned)