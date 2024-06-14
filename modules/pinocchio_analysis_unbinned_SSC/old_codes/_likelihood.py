import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/PySSC/')
import PySSC
import corner
import numpy as np
import matplotlib.pyplot as plt
import scipy
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
import analysis_mcmc_cosmo_samples as analysis
import pyccl as ccl
import h5py, glob
import emcee
import _planck_erosita_spt_samples as _samples

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
    pickle.dump(dat, open(filename, "wb"))  

#true cosmology
Omega_c_true = 0.30711 - 0.048254
Omega_b_true = 0.048254
sigma8_true = .8288
Omegam_true = 0.30711
True_value = [Omega_c_true + Omega_b_true, sigma8_true]

#define cosmology
cosmo = ccl.Cosmology(Omega_c = Omegam_true - Omega_b_true, Omega_b = Omega_b_true, 
                              h = 0.6777, sigma8 = sigma8_true, n_s=0.96)

code, index_analysis = sys.argv

sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC/mcmc_modules/')

likelihood = analysis.analysis_unbinned['likelihood'][int(index_analysis)]
analysis_name = analysis.analysis_unbinned['analysis_name'][int(index_analysis)]

if likelihood == 'full_unbinned_SSC' or likelihood == 'full_unbinned':
    import unbinned_model as ub
elif likelihood == 'hybrid_unbinned_SSC' or likelihood == 'hybrid_garrell_unbinned_SSC':
    import unbinned_model_hybrid as ub

mapping = ub.Mapping()
mapping.set_cosmology(cosmo)

#load data
fsky = analysis.fsky[analysis.analysis_name.index(analysis_name)]
mapping0 = ub.Mapping()
mapping0.set_cosmology(cosmo)

if analysis_name=='spt':
    redshift_edges = np.linspace(0.25, 1.5, 5)
elif analysis_name=='erosita':
    redshift_edges = np.linspace(0.01, 2, 5)
elif analysis_name=='planck':
    redshift_edges = np.linspace(0.01, 0.8, 5)
elif analysis_name=='pinocchio':
    redshift_edges = np.linspace(0.01, 1.2, 5)

redshift_intervals = [[redshift_edges[i], redshift_edges[i+1]] for i in range(len(redshift_edges)-1)]

if likelihood == 'hybrid_garrell_unbinned_SSC':
    
    default_cosmo_params = {'omega_b':cosmo['Omega_b']*cosmo['h']**2, 
                                'omega_cdm':cosmo['Omega_c']*cosmo['h']**2, 
                                'H0':cosmo['h']*100, 
                                'n_s':cosmo['n_s'], 
                                'sigma8': cosmo['sigma8'],
                                'output' : 'mPk'}

    z_arr = np.linspace(0.01,2.1,1000)
    nbins_T   = len(redshift_intervals)
    windows_T = np.zeros((nbins_T,len(z_arr)))
    for i, z_bin in enumerate(redshift_intervals):
        for k, z in enumerate(z_arr):
            if ((z>z_bin[0]) and (z<=z_bin[1])):
                windows_T[i,k] = 1  

    Sij_fullsky = PySSC.Sij_alt_fullsky(z_arr, windows_T, order=1, cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0)
    Sij = Sij_fullsky/fsky

z_grid = np.linspace(redshift_edges[0], redshift_edges[-1], 401)
logm_grid = np.linspace(14, 16, 400)

def mask_map():
    Z_grid = np.tile(z_grid,(len(logm_grid), 1))
    LogMvir_grid = np.tile(logm_grid,(len(z_grid), 1)).T
    mass_in = ccl.halos.massdef.MassDef('vir', 'matter')
    mass_out = ccl.halos.massdef.MassDef('500', 'critical')
    concentration = conc = ccl.halos.concentration.ConcentrationDuffy08(mass_def=mass_in)
    mass_ranslator = ccl.halos.massdef.mass_translator(mass_in=mass_in,
                                                       mass_out=mass_out, 
                                                       concentration=concentration)
    LogM500c_grid = np.zeros(Z_grid.shape)
    
    for k, z in enumerate(z_grid):
        M500c = np.array([mass_ranslator(cosmo, 10**LogM, 1/(1+z)) for LogM in logm_grid])
        LogM500c_grid[:,k] = np.log10(M500c)
    if analysis_name=='spt':
        mask = _samples.spt(Z_grid, 10**LogM500c_grid, 10**LogMvir_grid)
    elif analysis_name=='erosita':
        mask = _samples.erosita(Z_grid, 10**LogM500c_grid, 10**LogMvir_grid)
    elif analysis_name=='planck':
        mask = _samples.planck(Z_grid, 10**LogM500c_grid, 10**LogMvir_grid)
    elif analysis_name=='pinocchio':
        mask = _samples.pinocchio(Z_grid, 10**LogM500c_grid, 10**LogMvir_grid)
        
    return np.where(mask==True, 1, 0)

mask_sample = mask_map()

cat = np.load(analysis.analysis_name_file[analysis.analysis_name.index(analysis_name)], allow_pickle=True)
z_sample1, logm_sample1 = cat['z'], np.log10(cat['M'])

z_sample = z_sample1[(z_sample1 > z_grid[0])*(z_sample1 < z_grid[-1])*(logm_sample1 > logm_grid[0])*(logm_sample1 < logm_grid[-1])]
logm_sample = logm_sample1[(z_sample1 > z_grid[0])*(z_sample1 < z_grid[-1])*(logm_sample1 > logm_grid[0])*(logm_sample1 < logm_grid[-1])]

print(f'LIKELIHOOD= {likelihood}')
print(f'SAMPLE= {analysis_name}')
print(f'f_SKY= {fsky*100} %')
print(f'NCL= {len(z_sample):.0f} clusters')
print(len(z_sample), len(logm_sample))
#computing default amplitude of matter fluctuations

dN_dlogmdz_map = mapping0.compute_dN_dlogMdzdOmega_map(z_grid, logm_grid, fsky) * mask_sample
dN_dlogmdz_map1 = mapping0.compute_dN_dlogMdzdOmega_map(z_grid, logm_grid, fsky) 
halo_bias_map = mapping0.compute_halo_bias_map(z_grid, logm_grid) * mask_sample
Nth = mapping.compute_N_th(z_grid, logm_grid, dN_dlogmdz_map)
Nth1 = mapping.compute_N_th(z_grid, logm_grid, dN_dlogmdz_map1)
print(f'NTH = {Nth:.1f}')
print(Nth1)
if likelihood == 'full_unbinned_SSC' or likelihood == 'full_unbinned':
    sigma2_map = mapping0.compute_sigma2_map(z_grid, fsky)

def SSC_contribution(theta):
    Om, s8 = theta
    cosmo_new = ccl.Cosmology(Omega_c = Om - Omega_b_true, Omega_b = Omega_b_true, 
                              h = 0.6777, sigma8 = s8, n_s=0.96)
    mapping = ub.Mapping()
    mapping.set_cosmology(cosmo_new)
    dN_dlogmdz_map = mapping.compute_dN_dlogMdzdOmega_map(z_grid, logm_grid, fsky) * mask_sample
    Nth = mapping.compute_N_th(z_grid, logm_grid, dN_dlogmdz_map)
    
    if likelihood == 'full_unbinned':
        mapping.interp(z_grid, logm_grid, sigma2_map, dN_dlogmdz_map, halo_bias_map)
        dN_dzdlogM = mapping.N_map_interp_fct(logm_sample, z_sample, grid = False) 
        lnLnoSSC = standard_unbinned_likelihood.lnLikelihood_UnBinned_Poissonian(dN_dzdlogM, Nth)
        return lnLnoSSC
    
    if likelihood == 'hybrid_garrell_unbinned_SSC':
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
        return lnLnoSSC + lnLSSC

def loglikelihood(theta):
    Om, s8 = theta
    if Om > 1: return -np.inf
    if Om < 0.1: return -np.inf
    if s8 > 1: return -np.inf
    if s8 < 0.2: return -np.inf
    return SSC_contribution(theta)

nwalker = 70
ndim=2
initial = np.array([Omega_c_true + Omega_b_true, sigma8_true]) + .005*np.random.randn(1, ndim)
pos = np.array(initial) + .005*np.random.randn(nwalker, ndim)
print(loglikelihood([Omega_c_true + Omega_b_true, sigma8_true]))
sampler = emcee.EnsembleSampler(nwalker, ndim, loglikelihood,)
sampler.run_mcmc(pos, 200, progress=True);
flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)
name_save = f'/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC/SSC_contribution/SSC_mcmc/mcmc_{likelihood}_sample_{analysis_name}.pkl'
save_pickle(flat_samples, name_save)
