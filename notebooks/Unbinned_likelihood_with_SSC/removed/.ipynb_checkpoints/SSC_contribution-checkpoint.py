import sys
import corner
import numpy as np
import matplotlib.pyplot as plt
import scipy
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
import unbinned_likelihood
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import covariance as covar
import utils
import pandas as pd
import abundance as cl_count
import forecast
import emcee
import unbinned_model as ub
from lnlikelihood import lnLikelihood
import pyccl as ccl
import edit
import h5py, glob
Omega_c_true = 0.30711 - 0.048254
Omega_b_true = 0.048254
sigma8_true = .8288
Omegam_true = 0.30711
True_value = [Omega_c_true + Omega_b_true, sigma8_true]

cosmo = ccl.Cosmology(Omega_c = Omegam_true - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = sigma8_true, n_s=0.96)

mapping = ub.Mapping()
mapping.set_cosmology(cosmo)

code, name_analysis, index_analysis, SSC = sys.argv
print(sys.argv)
zmin, zmax = analysis.analysis[name_analysis]['redshift_bins'][int(index_analysis)]
logmmin, logmmax = analysis.analysis[name_analysis]['logm_bins'][int(index_analysis)]

z_range = [zmin, zmax]
logm_range = [logmmin, logmmax]

where_cat = '/sps/lsst/users/cpayerne/1000xsimulations/1000_simulations/afumagalli/catalogs/plc_14/*'
cat = glob.glob(where_cat)

for i ,c in enumerate(cat):
    cat_test = pd.read_csv(c ,sep=' ',skiprows=12, names=['M','z','dec','ra'])
    ra, dec, redshift, Mvir = cat_test['ra'], cat_test['dec'], cat_test['z'], cat_test['M']/0.6777
    if i == 3: break
    
mask = (redshift > z_range[0])&(redshift < z_range[1])
mask = mask &(np.log10(Mvir) > logm_range[0])&(np.log10(Mvir) < logm_range[1])
redshift_cut = redshift[mask]
Mvir_cut = Mvir[mask]
Nobs = len(Mvir_cut)
mapping = ub.Mapping()
z_sample = redshift_cut
logm_sample = np.log10(Mvir_cut)

mapping.create_reduced_sample(z_sample, logm_sample, 2000)

def SSC_contribution(Om):
    cosmo = ccl.Cosmology(Omega_c = Om - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = sigma8_true, n_s=0.96)
    
    mapping.set_cosmology(cosmo)
    z_grid = np.linspace(z_range[0], z_range[1], 510)
    logm_grid = np.linspace(logm_range[0], logm_range[1], 500)
    fsky = 1/4
    dN_dlogmdz_map = mapping.compute_dN_dlogMdzdOmega_map(z_grid, logm_grid, fsky)
    halo_bias_map = mapping.compute_halo_bias_map(z_grid, logm_grid)
    sigma2_map = mapping.compute_sigma2_map(z_grid, fsky)
    mapping.interp(z_grid, logm_grid, sigma2_map, dN_dlogmdz_map, halo_bias_map, )
    Nth = mapping.compute_N_th(z_grid, logm_grid, dN_dlogmdz_map)
    #
    NNSbb_thth = mapping.compute_NNSbb_thth(z_grid, logm_grid, sigma2_map, dN_dlogmdz_map, halo_bias_map, Nth)
    #
    NNSbb_obsobs = mapping.compute_NNSbb_obsobs(z_grid, logm_grid, 
                                        sigma2_map, dN_dlogmdz_map, halo_bias_map, 
                                        z_sample, logm_sample, Nobs, reduced_sample = True)
    #
    NNSbb_obsth = mapping.compute_NNSbb_obsth(z_grid, logm_grid, 
                                      sigma2_map, dN_dlogmdz_map, halo_bias_map, 
                                      z_sample, logm_sample, Nth, Nobs, reduced_sample = True)
    #
    NSb2_obs = mapping.compute_NSb2_obs(z_grid, logm_grid, sigma2_map, dN_dlogmdz_map, halo_bias_map, 
                            z_sample, logm_sample, Nth, Nobs, reduced_sample = True)
    #
    dN_dzdlogM = mapping.N_map_interp_fct(logm_sample, z_sample, grid = False)
    #
    lnLSSC = unbinned_likelihood.lnLikelihood_UnBinned_Poissonian_SSC(dN_dzdlogM, Nth, Nobs,
                                     NNSbb_obsobs,
                                     NNSbb_obsth,
                                     NNSbb_thth,
                                     NSb2_obs)
    #
    lnLnoSSC = unbinned_likelihood.lnLikelihood_UnBinned_Poissonian(dN_dzdlogM, Nth)
    return lnLnoSSC, lnLSSC-lnLnoSSC

Om = np.linspace(0.2, .5, 25)

res = []
for Om_ in Om:
    print(Om_)
    res.append(SSC_contribution(Om_))
    
np.save('SSC_contribution' + str(logmmin) + '_' + str(logmmax) + '.pkl', [Om, res])