import numpy as np

where_cat = '/sps/lsst/users/cpayerne/1000xsimulations/1000_simulations/afumagalli/catalogs/plc_14/*'
fsky = 1/4
resize = False
cat_number = 703

analysis_name = ['S1', 'S2', 'S3']
#redshift_bins = [[0.20, .25], [0.20, .30], [0.40, 0.45],]
#logm_bins =  [[14.30, 14.4], [14.30, 14.50], [14.30, 14.40],]
redshift_bins = [[0.20, .30], [0.20, .30], [0.2, 0.3],]
logm_bins =  [[14.30, 14.35], [14.30, 14.40], [14.30, 14.45],]
likelihood_name = ['full_unbinned_SSC', 'hybrid_unbinned_SSC', 'hybrid_garrell_unbinned_SSC']
likelihood_name = [ 'hybrid_garrell_unbinned_SSC' ]
likelihood_name = ['Gaussian_SSC_binned', 'Poisson_binned', ]
RZ = []
RM = []
NAME = []
LIKELIHOOD = []

for i in range(len(likelihood_name)):
    for j in range(len(analysis_name)):
        LIKELIHOOD.append(likelihood_name[i])
        NAME.append(analysis_name[j])
        RZ.append(redshift_bins[j])
        RM.append(logm_bins[j])

analysis_unbinned = {'redshift_interval':RZ, 'logm_interval':RM, 'analysis_name':NAME, 'likelihood':LIKELIHOOD}
#analysis_binned = {'nzbins':[20, 20, 20, 20], 'nmbins':[20, 20, 20, 20], 'redshift_interval':redshift_bins, 'logm_interval':logm_bins}
