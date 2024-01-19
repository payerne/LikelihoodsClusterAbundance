import corner
import numpy as np
import matplotlib.pyplot as plt
import scipy
import emcee
import glob
import pyccl as ccl
import pandas as pd
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
import analysis
import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules')
import covariance

sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC')
import unbinned_model as ub
import unbinned_likelihood

Omega_c_true = 0.30711 - 0.048254
Omega_b_true = 0.048254
sigma8_true = .8288
Omegam_true = 0.30711
True_value = [Omega_c_true + Omega_b_true, sigma8_true]

code, index_analysis = sys.argv
zmin, zmax = analysis.analysis[index_analysis]['redshift_bins']
name_analysis = analysis.analysis[index_analysis]['name']
likelihood = analysis.analysis[index_analysis]['likelihood']
logmmin, logmmax = analysis.analysis[index_analysis]['logm_bins']
catalog_pinocchio_name = analysis.analysis[index_analysis]['cat_name']
wheretosave = analysis.analysis[index_analysis]['where_to_save']

cat_test = pd.read_csv(catalog_pinocchio_name ,sep=' ',skiprows=12, names=['M','z','dec','ra'])
ra, dec, redshift, Mvir = cat_test['ra'], cat_test['dec'], cat_test['z'], cat_test['M']/0.6777

mask = (redshift > zmin)&(redshift < zmax)
mask = mask &(np.log10(Mvir) > logmmin)&(np.log10(Mvir) < logmmax)
redshift_cut = redshift[mask]
Mvir_cut = Mvir[mask]
print(len(Mvir_cut))
Nobs = len(Mvir_cut)
print(Nobs)

z_sample = redshift_cut
logm_sample = np.log10(Mvir_cut)

z_grid = np.linspace(zmin, zmax, 310)
logm_grid = np.linspace(logmmin, logmmax, 300)
fsky = 1/4

cosmo = ccl.Cosmology(Omega_c = Omegam_true - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = sigma8_true, n_s=0.96)
mapping = ub.Mapping()
mapping.set_cosmology(cosmo)
dN_dlogmdz_map = mapping.compute_dN_dlogMdzdOmega_map(z_grid, logm_grid, fsky)
sigma2_map = mapping.compute_sigma2_map(z_grid, fsky)
halo_bias_map0 = mapping.compute_halo_bias_map(z_grid, logm_grid)
mapping.interp(z_grid, logm_grid, sigma2_map, dN_dlogmdz_map, halo_bias_map0)
mapping.create_reduced_sample(z_sample, logm_sample, 2000)

#Binned SSC contribution
covar = covariance.Covariance_matrix()
Sij = covar.matter_fluctuation_amplitude_fullsky([[zmin, zmax]])

def lnL_full(p):
    
    Om, s8 = p
    
    if (Om > 1): return -np.inf
    if (Om < 0.2): return -np.inf
    if (s8 > 1): return -np.inf
    if (s8 < 0.4): return -np.inf
    #if (h > 1): return -np.inf
    #if (h < 0.5): return -np.inf
    #if (n_s > 1): return -np.inf
    #if (n_s < 0.9): return -np.inf

    cosmo_new = ccl.Cosmology(Omega_c = Om - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = s8, n_s=0.96)
    
    mapping.set_cosmology(cosmo_new)
    
    dN_dlogmdz_map = mapping.compute_dN_dlogMdzdOmega_map(z_grid, logm_grid, fsky)
    
    if likelihood=='unbinnedSSC':

        halo_bias_map = mapping.compute_halo_bias_map(z_grid, logm_grid)

        mapping.interp(z_grid, logm_grid, sigma2_map, dN_dlogmdz_map, halo_bias_map, )
        
        Nth = mapping.compute_N_th(z_grid, logm_grid, dN_dlogmdz_map)
        #1
        NNSbb_thth = mapping.compute_NNSbb_thth(z_grid, logm_grid, sigma2_map, dN_dlogmdz_map, halo_bias_map, Nth)
        #2
        NNSbb_obsobs = mapping.compute_NNSbb_obsobs(z_grid, logm_grid, 
                                            sigma2_map, dN_dlogmdz_map, halo_bias_map, 
                                            z_sample, logm_sample, Nobs, reduced_sample = True)
        #3
        NNSbb_obsth = mapping.compute_NNSbb_obsth(z_grid, logm_grid, 
                                          sigma2_map, dN_dlogmdz_map, halo_bias_map, 
                                          z_sample, logm_sample, Nth, Nobs, reduced_sample = True)
        #4
        NSb2_obs = mapping.compute_NSb2_obs(z_grid, logm_grid, sigma2_map, dN_dlogmdz_map, halo_bias_map, 
                                z_sample, logm_sample, Nth, Nobs, reduced_sample = True)
        
        dN_dzdlogM = mapping.N_map_interp_fct(logm_sample, z_sample, grid = False)
        
        lnL_SSC = unbinned_likelihood.lnLikelihood_UnBinned_Poissonian_SSC(dN_dzdlogM, Nth, Nobs,
                                                                             NNSbb_obsobs,
                                                                             NNSbb_obsth,
                                                                             NNSbb_thth,
                                                                             NSb2_obs)
        
        lnL_Poisson = unbinned_likelihood.lnLikelihood_UnBinned_Poissonian(dN_dzdlogM, Nth)
        
        lnL_ = lnL_SSC
        
    elif likelihood == 'unbinnedstd': 
        
        mapping.interp(z_grid, logm_grid, sigma2_map, dN_dlogmdz_map, halo_bias_map0)
        
        Nth = mapping.compute_N_th(z_grid, logm_grid, dN_dlogmdz_map)
        
        dN_dzdlogM = mapping.N_map_interp_fct(logm_sample, z_sample, grid = False)
        
        lnL_ = unbinned_likelihood.lnLikelihood_UnBinned_Poissonian(dN_dzdlogM, Nth)
        
    elif likelihood == 'P18': 
        
        halo_bias_map = mapping.compute_halo_bias_map(z_grid, logm_grid)
        
        mapping.interp(z_grid, logm_grid, sigma2_map, dN_dlogmdz_map, halo_bias_map)
        
        Nth = mapping.compute_N_th(z_grid, logm_grid, dN_dlogmdz_map)
        
        dN_dzdlogM = mapping.N_map_interp_fct(logm_sample, z_sample, grid = False)
        
        lnL_ = unbinned_likelihood.lnLikelihood_UnBinned_Poissonian(dN_dzdlogM, Nth)
        
        #binned likelihood
        
        Naverage_bias = np.trapz(np.trapz(halo_bias_map * dN_dlogmdz_map, z_grid, axis=1), logm_grid, axis = 0)
        
        average_bias = Naverage_bias/np.trapz(np.trapz (dN_dlogmdz_map, z_grid, axis=1), logm_grid, axis = 0)
        
        sigma2_SSC = Nth ** 2 * average_bias ** 2 * Sij[0]/fsky
        
        sigma2 = Nth + sigma2_SSC
        
        lnL_Binned = -0.5*((Nth - Nobs)/np.sqrt(sigma2))**2 - np.log(sigma2**.5)
        
        lnL_  = lnL_ + lnL_Binned
        
        lnL_ = lnL_[0]
        
    return lnL_

import time
t = time.time()
true = [0.30711, .8288]
print(lnL_full(true))
tf = time.time()
print(tf-t)

#mcmc
nwalkers = 100
initial = np.random.randn(nwalkers, len(true))*.001 + np.array(np.array(true))
sampler = emcee.EnsembleSampler(nwalkers, len(true), lnL_full,)
sampler.run_mcmc(initial, 150, progress=True)
f = sampler.get_chain(discard=0, thin=1, flat=True)
np.save(wheretosave + '_' + name_analysis, f)