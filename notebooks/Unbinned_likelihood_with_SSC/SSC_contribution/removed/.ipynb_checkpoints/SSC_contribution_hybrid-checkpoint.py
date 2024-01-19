import sys
import PINOCCHIO_cat
sys.path.append('/pbs/throng/lsst/users/cpayerne/PySSC/')
import PySSC
import corner
import numpy as np
import matplotlib.pyplot as plt
import scipy
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
import analysis

sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import covariance as covar
import utils
import pandas as pd
import abundance as cl_count
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC/mcmc_modules/')
import unbinned_likelihood
import unbinned_model_hybrid as ub
import pickle

def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()

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

code, index_analysis = sys.argv
print(sys.argv)
zmin, zmax = analysis.analysis_unbinned['redshift_interval'][int(index_analysis)]
logmmin, logmmax = analysis.analysis_unbinned['logm_interval'][int(index_analysis)]
print(zmin, zmax)
print(logmmin, logmmax)
z_range = [zmin, zmax]
logm_range = [logmmin, logmmax]
print(z_range, logm_range)

where_cat = analysis.where_cat
fsky = analysis.fsky
cat = analysis.cat_number
ra, dec, redshift, Mvir = PINOCCHIO_cat.catalog(where_cat, cat, fsky, resize=analysis.resize)
print(len(ra))

mask = (redshift > z_range[0])&(redshift < z_range[1])
mask = mask &(np.log10(Mvir) > logm_range[0])&(np.log10(Mvir) < logm_range[1])
redshift_cut = redshift[mask]
Mvir_cut = Mvir[mask]
Nobs = len(Mvir_cut)
z_sample = redshift_cut
logm_sample = np.array(np.log10(Mvir_cut))
print(len(z_sample))

redshift_edges = np.linspace(z_range[0], z_range[1], 6)
redshift_intervals = [[redshift_edges[i], redshift_edges[i+1]] for i in range(len(redshift_edges)-1)]
mapping0 = ub.Mapping()
mapping0.set_cosmology(cosmo)
z_grid = np.linspace(z_range[0], z_range[1], 2000)
logm_grid = np.linspace(logm_range[0], logm_range[1], 500)
#fsky = 1/4
dN_dlogmdz_map = mapping0.compute_dN_dlogMdzdOmega_map(z_grid, logm_grid, fsky)
halo_bias_map = mapping0.compute_halo_bias_map(z_grid, logm_grid)
Sij = mapping.compute_Sij_map(redshift_intervals, fsky)
print(Nobs)
Nth = mapping0.compute_N_th(z_grid, logm_grid, dN_dlogmdz_map)
print(Nth)
def SSC_contribution(Om, s8):
    cosmo_new = ccl.Cosmology(Omega_c = Om - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = s8, n_s=0.96)
    mapping = ub.Mapping()
    mapping.set_cosmology(cosmo_new)
    dN_dlogmdz_map = mapping.compute_dN_dlogMdzdOmega_map(z_grid, logm_grid, fsky)
    #
    mapping.compute_bdNdm_zbins_and_dNdm_zbins(z_grid, logm_grid, dN_dlogmdz_map, halo_bias_map, redshift_intervals, fsky)
    mapping.compute_Nb_zbins(z_grid, logm_grid, dN_dlogmdz_map, halo_bias_map, redshift_intervals, fsky)
    mapping.interp(z_grid, logm_grid, dN_dlogmdz_map, halo_bias_map, redshift_intervals, Sij, fsky)
    #
    NNSbb_thth = mapping.compute_NNSbb_thth(z_grid, logm_grid, dN_dlogmdz_map, halo_bias_map, redshift_intervals, Sij, fsky)
    #
    NNSbb_obsth = mapping.compute_NNSbb_thobs(z_grid, logm_grid, dN_dlogmdz_map, halo_bias_map, redshift_intervals, Sij, fsky, z_sample, logm_sample)
    #
    NNSbb_obsobs = mapping.compute_NNSbb_obsobs(z_grid, logm_grid, dN_dlogmdz_map, halo_bias_map, redshift_intervals, Sij, fsky, z_sample, logm_sample)
    #
    NSb2_obs = mapping.compute_NSb2_obs(z_grid, logm_grid, dN_dlogmdz_map, halo_bias_map, redshift_intervals, Sij, fsky, z_sample, logm_sample)
    #
    dN_dlogM = mapping.N_map_interp_fct(z_sample, logm_sample, redshift_intervals)
    #
    Nth = mapping.compute_N_th(z_grid, logm_grid, dN_dlogmdz_map)
    print(Nth)
    #
    #lnLSSC = unbinned_likelihood.lnLikelihood_UnBinned_Poissonian_SSC(dN_dlogM, Nth, Nobs,
    #                                                                             NNSbb_obsobs,
    #                                                                             NNSbb_obsth,
    #                                                                             NNSbb_thth,
    #                                                                             NSb2_obs)
    #
    lnFSSC = mapping.compute_full_SSC(z_grid, logm_grid, dN_dlogmdz_map, halo_bias_map, redshift_intervals, Sij, fsky, z_sample, logm_sample)
    #
    lnLnoSSC = unbinned_likelihood.lnLikelihood_UnBinned_Poissonian(dN_dlogM, Nth)
    #return lnLSSC, lnLnoSSC, lnLSSC-lnLnoSSC, Nth, NNSbb_thth, NNSbb_obsobs, NNSbb_obsth, NSb2_obs
    return lnLnoSSC + lnFSSC, lnLnoSSC, 1, 1, 1, 1, 1, 1

Om = np.linspace(0.2, .4, 300)
s8 = np.linspace(0.6, 1, 35)

Om_tab = True
if Om_tab == True:
    res = []
    for Om_ in Om:
        res.append(SSC_contribution(Om_,sigma8_true))
    save_pickle([Om, res], f'/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC/SSC_contribution/SSC/Unbinned_hybrid_cat={cat}_Nobs={Nobs}_z[{zmin:.2f},{zmax:.2f}]_logm[{logmmin:.2f},{logmmax:.2f}]_fsky[{fsky:.4f}].pkl')

# s8_tab = False
# if s8_tab == True:
#    res = []
#    for s8_ in s8:
#        res.append(SSC_contribution(Omegam_true,s8_))
#    save_pickle([s8, res], f's8_SSC_contribution_' + {logmmin:.2f} + '_' + {logmmax:.2f} + '.pkl')
