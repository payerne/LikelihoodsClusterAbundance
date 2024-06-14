import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/PySSC/')
import PySSC
import corner
import numpy as np
import matplotlib.pyplot as plt
import scipy
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
import _analysis_mock_samples as analysis
import pyccl as ccl
import h5py, glob
import emcee

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

code, likelihood, analysis_case, index_analysis = sys.argv

sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC/mcmc_modules/')
if likelihood == 'standard_unbinned':
    import unbinned_model as ub
elif likelihood == 'hybrid_garrell':
    import unbinned_model_hybrid as ub

mapping = ub.Mapping()
mapping.set_cosmology(cosmo)
analysis_case_dict = analysis.which_sample(analysis_case)
zmin, zmax = analysis_case_dict['redshift_interval'][int(index_analysis)]
logmmin, logmmax = analysis_case_dict['logm_interval'][int(index_analysis)]
nbins_bybrid = analysis_case_dict['number_bins_hybrid'][int(index_analysis)]
analysis_name = analysis_case_dict['analysis_name'][int(index_analysis)]
print(nbins_bybrid)
print(zmin, zmax)
print(logmmin, logmmax)
print(analysis_name)
z_range = [zmin, zmax]
logm_range = [logmmin, logmmax]
print(z_range, logm_range)

#load data
where_cat = analysis.where_cat
fsky = analysis.fsky
cat = analysis.cat_number
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
#dN_dlogmdz_map = mapping0.compute_dN_dlogMdzdOmega_map(z_grid, logm_grid, fsky)
halo_bias_map = mapping0.compute_halo_bias_map(z_grid, logm_grid)

#computing default amplitude of matter fluctuations
if likelihood == 'standard_unbinned':
    sigma2_map = mapping0.compute_sigma2_map(z_grid, fsky)
if likelihood == 'hybrid_garrell':
    redshift_edges = np.linspace(z_range[0], z_range[1], nbins_bybrid + 1)
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
    
    if likelihood == 'standard_unbinned':
        mapping.interp(z_grid, logm_grid, sigma2_map, dN_dlogmdz_map, halo_bias_map)
        dN_dzdlogM = mapping.N_map_interp_fct(logm_sample, z_sample, grid = False)
        lnLnoSSC = standard_unbinned_likelihood.lnLikelihood_UnBinned_Poissonian(dN_dzdlogM, Nth)
        print(lnLnoSSC)
        return lnLnoSSC, 1, 1, 1, 1, 1, 1, 1
    
    
    if likelihood == 'hybrid_garrell':
        mapping.compute_bdNdm_zbins_and_dNdm_zbins(z_grid, logm_grid, 
                                                   dN_dlogmdz_map, halo_bias_map, 
                                                   redshift_intervals, fsky)
        mapping.compute_Nb_zbins(z_grid, logm_grid, 
                                 dN_dlogmdz_map, halo_bias_map, 
                                 redshift_intervals, fsky)
        mapping.interp(z_grid, logm_grid, dN_dlogmdz_map, halo_bias_map, 
                       redshift_intervals, Sij, fsky)
        dN_dlogM = mapping.N_map_interp_fct(z_sample, logm_sample, redshift_intervals)
        lnLSSC = mapping.compute_full_SSC_hybrid_garrell(z_grid, logm_grid, 
                                                             dN_dlogmdz_map, halo_bias_map, 
                                                             redshift_intervals, Sij, fsky, 
                                                             z_sample, logm_sample)
        lnLnoSSC = standard_unbinned_likelihood.lnLikelihood_UnBinned_Poissonian(dN_dlogM, Nth)
        
        return lnLnoSSC + lnLSSC, lnLnoSSC, 1, 1, 1, 1, 1, 1

mcmc=False

if mcmc==False:

    ww = SSC_contribution(Omega_c_true + Omega_b_true, sigma8_true)
    Om = np.linspace(0.2, .5, 1000)
    s8 = np.linspace(0.6, 1, 35)
    name_save = f'/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC/SSC_contribution/SSC_paper/{likelihood}_cat={cat}_sample_{analysis_name}.pkl'
    Om_tab = True
    if Om_tab == True:
        res = []
        for Om_ in Om:
            print(1)
            res.append(SSC_contribution(Om_,sigma8_true))
        save_pickle([Om, np.array(res)], name_save)
else: 
    def loglikelihood(theta):
        Om, s8 = theta
        if Om > 1: return -np.inf
        if Om < 0.1: return -np.inf
        if s8 > 1: return -np.inf
        if s8 < 0.2: return -np.inf
        return - SSC_contribution(Om, s8)[0]
    nwalker = 70
    ndim=2
    initial = np.array([Omega_c_true + Omega_b_true, sigma8_true]) + .005*np.random.randn(1, ndim)
    pos = np.array(initial) + .005*np.random.randn(nwalker, ndim)
    sampler = emcee.EnsembleSampler(nwalker, ndim, loglikelihood,)
    sampler.run_mcmc(pos, 50, progress=True);
    flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)
    name_save = f'/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC/SSC_contribution/SSC_paper/mcmc_{likelihood}_cat={cat}_sample_{analysis_name}.pkl'
    save_pickle(flat_samples, name_save)
