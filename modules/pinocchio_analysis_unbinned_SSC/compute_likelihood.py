import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/PySSC/')
import PySSC
import corner
import numpy as np
import matplotlib.pyplot as plt
import scipy
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
import analysis
import pyccl as ccl
import h5py, glob

sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import covariance as covar
import utils
import pandas as pd
import abundance as cl_count

#import unbinned module
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC/mcmc_modules/')
import standard_unbinned_likelihood
import PINOCCHIO_cat
import pickle

def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()

#true cosmology
Omega_c_true = 0.30711 - 0.048254
Omega_b_true = 0.048254
sigma8_true = .8288
Omegam_true = 0.30711
True_value = [Omega_c_true + Omega_b_true, sigma8_true]

#define cosmology
cosmo = ccl.Cosmology(Omega_c = Omegam_true - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = sigma8_true, n_s=0.96)

code, index_analysis = sys.argv

sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC/mcmc_modules/')
likelihood = analysis.analysis_unbinned['likelihood'][int(index_analysis)]
if likelihood == 'full_unbinned_SSC' or likelihood == 'full_unbinned':
    import unbinned_model as ub
elif likelihood == 'hybrid_unbinned_SSC' or likelihood == 'hybrid_garrell_unbinned_SSC':
    import unbinned_model_hybrid as ub
    
mapping = ub.Mapping()
mapping.set_cosmology(cosmo)
zmin, zmax = analysis.analysis_unbinned['redshift_interval'][int(index_analysis)]
logmmin, logmmax = analysis.analysis_unbinned['logm_interval'][int(index_analysis)]
print(zmin, zmax)
print(logmmin, logmmax)
z_range = [zmin, zmax]
logm_range = [logmmin, logmmax]
print(z_range, logm_range)

#load data
where_cat = analysis.where_cat
fsky = analysis.fsky
cat = analysis.cat_number
analysis_name = analysis.analysis_unbinned['analysis_name'][int(index_analysis)]
ra, dec, redshift, Mvir = PINOCCHIO_cat.catalog(where_cat, cat, fsky, resize=analysis.resize)
mask = (redshift > z_range[0])&(redshift < z_range[1])
mask = mask &(np.log10(Mvir) > logm_range[0])&(np.log10(Mvir) < logm_range[1])
redshift_cut = redshift[mask]
Mvir_cut = Mvir[mask]
Nobs = len(Mvir_cut)
z_sample = redshift_cut
logm_sample = np.array(np.log10(Mvir_cut))
print(f'sample = {len(z_sample)} clusters')

mapping0 = ub.Mapping()
mapping0.set_cosmology(cosmo)
z_grid = np.linspace(z_range[0], z_range[1], 1000)
logm_grid = np.linspace(logm_range[0], logm_range[1], 500)
dN_dlogmdz_map = mapping0.compute_dN_dlogMdzdOmega_map(z_grid, logm_grid, fsky)
halo_bias_map = mapping0.compute_halo_bias_map(z_grid, logm_grid)

#computing default amplitude of matter fluctuations
if likelihood == 'full_unbinned_SSC' or likelihood == 'full_unbinned':
    sigma2_map = mapping0.compute_sigma2_map(z_grid, fsky)
if (likelihood == 'hybrid_unbinned_SSC') or (likelihood == 'hybrid_garrell_unbinned_SSC'):
    redshift_edges = np.linspace(z_range[0], z_range[1], 4)
    redshift_intervals = [[redshift_edges[i], redshift_edges[i+1]] for i in range(len(redshift_edges)-1)]
    Sij = mapping.compute_Sij_map(redshift_intervals, fsky)

reduced_sample = False

def SSC_contribution(Om, s8):
    cosmo_new = ccl.Cosmology(Omega_c = Om - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = s8, n_s=0.96)
    mapping = ub.Mapping()
    mapping.set_cosmology(cosmo_new)
    dN_dlogmdz_map = mapping.compute_dN_dlogMdzdOmega_map(z_grid, logm_grid, fsky)
    Nth = mapping.compute_N_th(z_grid, logm_grid, dN_dlogmdz_map)
    
    if likelihood == 'full_unbinned':
        
        mapping.interp(z_grid, logm_grid, sigma2_map, dN_dlogmdz_map, halo_bias_map)
        dN_dzdlogM = mapping.N_map_interp_fct(logm_sample, z_sample, grid = False)
        lnLnoSSC = standard_unbinned_likelihood.lnLikelihood_UnBinned_Poissonian(dN_dzdlogM, Nth)
        return lnLnoSSC, lnLnoSSC, 1, 1, 1, 1, 1, 1
    
    if likelihood == 'full_unbinned_SSC':
        
        mapping.interp(z_grid, logm_grid, sigma2_map, dN_dlogmdz_map, halo_bias_map)
        #Nth = mapping.compute_N_th(z_grid, logm_grid, dN_dlogmdz_map)
        NNSbb_thth = mapping.compute_NNSbb_thth(z_grid, logm_grid, 
                                                sigma2_map, dN_dlogmdz_map, halo_bias_map, 
                                                Nth)
        NNSbb_obsobs = mapping.compute_NNSbb_obsobs(z_grid, logm_grid, 
                                                    sigma2_map, dN_dlogmdz_map, halo_bias_map, 
                                                    z_sample, logm_sample, 
                                                    Nobs, reduced_sample = reduced_sample)
        NNSbb_obsth = mapping.compute_NNSbb_obsth(z_grid, logm_grid, 
                                                  sigma2_map, dN_dlogmdz_map, halo_bias_map, 
                                                  z_sample, logm_sample, 
                                                  Nth, Nobs, 
                                                  reduced_sample = reduced_sample)
        NSb2_obs = mapping.compute_NSb2_obs(z_grid, logm_grid, 
                                            sigma2_map, dN_dlogmdz_map, halo_bias_map, 
                                            z_sample, logm_sample, 
                                            Nth, Nobs, 
                                            reduced_sample = reduced_sample)
        dN_dzdlogM = mapping.N_map_interp_fct(logm_sample, z_sample, grid = False)
        lnLSSC = mapping.compute_W_SSC(NNSbb_obsobs, NNSbb_obsth, NNSbb_thth, NSb2_obs)
        lnLnoSSC = standard_unbinned_likelihood.lnLikelihood_UnBinned_Poissonian(dN_dzdlogM, Nth)
        return lnLSSC+lnLnoSSC, lnLnoSSC, lnLSSC-lnLnoSSC, Nth, NNSbb_thth, NNSbb_obsobs, NNSbb_obsth, NSb2_obs
    
    if likelihood == 'hybrid_unbinned_SSC' or likelihood == 'hybrid_garrell_unbinned_SSC':
        
        mapping.compute_bdNdm_zbins_and_dNdm_zbins(z_grid, logm_grid, 
                                                   dN_dlogmdz_map, halo_bias_map, 
                                                   redshift_intervals, fsky)
        mapping.compute_Nb_zbins(z_grid, logm_grid, 
                                 dN_dlogmdz_map, halo_bias_map, 
                                 redshift_intervals, fsky)
        mapping.interp(z_grid, logm_grid, dN_dlogmdz_map, halo_bias_map, 
                       redshift_intervals, Sij, fsky)
        dN_dlogM = mapping.N_map_interp_fct(z_sample, logm_sample, redshift_intervals)
        
        if likelihood == 'hybrid_unbinned_SSC':
            lnLSSC = mapping.compute_full_SSC_hybrid(z_grid, logm_grid, 
                                          dN_dlogmdz_map, halo_bias_map, 
                                          redshift_intervals, Sij, fsky,
                                          z_sample, logm_sample)
            
        if likelihood == 'hybrid_garrell_unbinned_SSC':
            lnLSSC = mapping.compute_full_SSC_hybrid_garrell(z_grid, logm_grid, 
                                                             dN_dlogmdz_map, halo_bias_map, 
                                                             redshift_intervals, Sij, fsky, 
                                                             z_sample, logm_sample)
            
        lnLnoSSC = standard_unbinned_likelihood.lnLikelihood_UnBinned_Poissonian(dN_dlogM, Nth)
        
        return lnLnoSSC + lnLSSC, lnLnoSSC, 1, 1, 1, 1, 1, 1

ww = SSC_contribution(Omega_c_true + Omega_b_true, sigma8_true)
Om = np.linspace(0.2, .4, 150)
s8 = np.linspace(0.6, 1, 35)
#Nobs={Nobs}_z[{zmin:.2f},{zmax:.2f}]_logm[{logmmin:.2f},{logmmax:.2f}]_fsky[{fsky:.4f}]
name_save = f'/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC/SSC_contribution/SSC/{likelihood}_cat={cat}_sample_{analysis_name}.pkl'
Om_tab = True
if Om_tab == True:
    res = []
    for Om_ in Om:
        print(1)
        res.append(SSC_contribution(Om_,sigma8_true))
    save_pickle([Om, np.array(res)], name_save)
