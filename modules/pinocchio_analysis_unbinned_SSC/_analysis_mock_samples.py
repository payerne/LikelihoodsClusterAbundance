import numpy as np
import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import PINOCCHIO_cat

where_cat = '/sps/lsst/users/cpayerne/1000xsimulations/1000_simulations/afumagalli/catalogs/plc_14/*'
fsky = 1/4
resize = False
cat_number = 400
#redshift_bins = [[0.20, .25], [0.20, .30], [0.40, 0.45],]
#logm_bins =  [[14.30, 14.4], [14.30, 14.50], [14.30, 14.40],]

#case = 'standard_unbinned'


def which_sample(case):

    if case == 'paper':

        analysis_name = ['S1', 'S2', 'S3', 'S4', 'S5','S6','S7','S8']
        number_bins_hybrid = [3,3,3,3,3,3,3,3]
        logm_bins =  [ [14.2, 15.6], [14.2, 15.6], [14.2, 15.6], [14.2, 15.6], [14.2, 15.6], [14.2, 15.6],  [14.2, 15.6],  [14.2, 15.6]]
        redshift_bins = [ [0.2, .4],  [0.2, .5],  [0.2, .6],  [0.2, .7],  [0.2, .8],  [0.2, .9],  [0.2, 1.],  [0.2, 1.2]]

        analysis_name += ['S11', 'S12', 'S13', 'S14', 'S15','S16','S17','S18','S19','S20']
        number_bins_hybrid += [3,3,3,3,3,3,3,3,3,3]
        logm_bins += [ [14.2, 15.6], [14.3, 15.6], [14.4, 15.6], [14.5, 15.6], [14.6, 15.6], [14.7, 15.6], [14.8, 15.6], [14.9, 15.6],  [15, 15.6], [15.1, 15.6]]
        redshift_bins += [[0.2, 1.2], [0.2, 1.2],  [0.2, 1.2], [0.2, 1.2], [0.2, 1.2], [0.2, 1.2], [0.2, 1.2], [0.2, 1.2], [0.2, 1.2], [0.2, 1.2]]

    if case =='varying_nbins_hybrid':

        analysis_name = ['S1_varying_nbins_hybrid', 'S2_varying_nbins_hybrid', 'S3_varying_nbins_hybrid',
                         'S4_varying_nbins_hybrid', 'S5_varying_nbins_hybrid','S6_varying_nbins_hybrid',
                         'S7_varying_nbins_hybrid','S8_varying_nbins_hybrid','S9_varying_nbins_hybrid',
                         'S10_varying_nbins_hybrid']
        number_bins_hybrid = [1,2,3,4,5,6,7,8,9,10]
        logm_bins =  [ [14.2, 15.6] for i in range(10)]
        redshift_bins = [ [0.2, 1.2] for i in range(10)]

    if case =='standard_unbinned':

        analysis_name = ['S1_standard_unbinned']
        number_bins_hybrid = [3]
        logm_bins =  [[14.2, 15.6]]
        redshift_bins = [[0.2, 1.2]]

    N = []
    ra, dec, redshift, Mvir = PINOCCHIO_cat.catalog(where_cat, cat_number, fsky, resize=False)
    for i in range(len(logm_bins)):
        mask = (redshift > redshift_bins[i][0])&(redshift < redshift_bins[i][1])
        mask = mask &(np.log10(Mvir) > logm_bins[i][0])&(np.log10(Mvir) < logm_bins[i][1])
        Mvir_cut = Mvir[mask]
        Nobs = len(Mvir_cut)
        N.append(Nobs)
    RZ = []
    RM = []
    NAME = []
    NBIN_HYBRID = []

    for j in range(len(analysis_name)):
        NAME.append(analysis_name[j])
        RZ.append(redshift_bins[j])
        RM.append(logm_bins[j])
        NBIN_HYBRID.append(number_bins_hybrid[j])

    analysis_unbinned = {'N_CLUSTERS': N, 'redshift_interval':RZ, 'logm_interval':RM, 'analysis_name':NAME, 'number_bins_hybrid':NBIN_HYBRID}
    return analysis_unbinned
#analysis_binned = {'nzbins':[20, 20, 20, 20], 'nmbins':[20, 20, 20, 20], 'redshift_interval':redshift_bins, 'logm_interval':logm_bins}
