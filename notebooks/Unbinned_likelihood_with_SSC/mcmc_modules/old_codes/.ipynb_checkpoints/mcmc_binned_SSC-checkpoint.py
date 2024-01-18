import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
import analysis
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import covariance as covar
import utils
import pandas as pd
import abundance as cl_count
import forecast
import emcee
from lnlikelihood import lnLikelihood
import pyccl as ccl
import edit
import h5py, glob
Omega_c_true = 0.30711 - 0.048254
Omega_b_true = 0.048254
sigma8_true = .8288
Omegam_true = 0.30711
True_value = [Omega_c_true + Omega_b_true, sigma8_true]

where_cat = '/sps/lsst/users/cpayerne/1000xsimulations/1000_simulations/afumagalli/catalogs/plc_14/*'
cat = glob.glob(where_cat)

code, index_analysis, SSC = sys.argv
print(sys.argv)
index_bin_in_analysis = 0
zmin, zmax = analysis.analysis[index_analysis]['redshift_bins'][int(index_bin_in_analysis)]
name_analysis = analysis.analysis[index_analysis]['name']
logmmin, logmmax = analysis.analysis[index_analysis]['logm_bins'][int(index_bin_in_analysis)]
wheretosave = analysis.analysis[index_analysis]['where_to_save']
print(zmin, zmax)
print(logmmin, logmmax)
print(name_analysis)
for i ,c in enumerate(cat):
    cat_test = pd.read_csv(c ,sep=' ',skiprows=12, names=['M','z','dec','ra'])
    ra, dec, redshift, Mvir = cat_test['ra'], cat_test['dec'], cat_test['z'], cat_test['M']/0.6777
    if i == 20: break

mask = (redshift > zmin)&(redshift < zmax)
mask = mask &(np.log10(Mvir) > logmmin)&(np.log10(Mvir) < logmmax)
redshift_cut = redshift[mask]
Mvir_cut = Mvir[mask]

lnL = lnLikelihood()
clc = cl_count.ClusterAbundance()
clc.sky_area = (0.25)*4*np.pi
clc.f_sky = clc.sky_area/(4*np.pi)
z_grid = np.linspace(zmin, zmax, 500)
logm_grid = np.linspace(logmmin, logmmax, 501)

z_corner = np.linspace(zmin, zmax, 5)
log10m_corner = np.linspace(logmmin, logmmax, 5)
Z_bin = [[z_corner[i], z_corner[i+1]] for i in range(len(z_corner)-1)]
LogMass_bin = [[log10m_corner[i], log10m_corner[i+1]] for i in range(len(log10m_corner)-1)]
#print(LogMass_bin)
Nobs, a, b = np.histogram2d(redshift_cut, np.log10(Mvir_cut), bins = [z_corner, log10m_corner])
print(Nobs)

if SSC == 'SSC':
    #choose the halo mass function and mass definition
    cosmo = ccl.Cosmology(Omega_c = Omega_c_true + Omega_b_true - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = sigma8_true, n_s=0.96)
    massdef = ccl.halos.massdef.MassDef('vir', 'critical', c_m_relation=None)
    hmd = ccl.halos.hmfunc.MassFuncDespali16(cosmo, mass_def=massdef)
    halobias = ccl.halos.hbias.HaloBiasTinker10(cosmo, mass_def= massdef, mass_def_strict=True)
    clc.set_cosmology(cosmo = cosmo, hmd = hmd, massdef = massdef)
    clc.sky_area = (0.25) * 4 * np.pi
    clc.f_sky = clc.sky_area/(4*np.pi)
    #z_grid = np.linspace(0., 2.1, 2500)
    #logm_grid = np.linspace(14.1, 15.7, 2501)
    clc.compute_multiplicity_grid_MZ(z_grid = z_grid, logm_grid = logm_grid)
    Abundance = clc.Cluster_Abundance_MZ(Redshift_bin = Z_bin, Proxy_bin = LogMass_bin, method = 'simps')
    clc.compute_halo_bias_grid_MZ(z_grid = z_corner, logm_grid = log10m_corner, halobiais = halobias)
    NHalo_bias = clc.Nhalo_bias_MZ(Redshift_bin = Z_bin, Proxy_bin = LogMass_bin, method = 'simps')
    Halo_bias = NHalo_bias/Abundance
    Covariance = covar.Covariance_matrix()
    Sij_fullsky = Covariance.matter_fluctuation_amplitude_fullsky(Z_bin)
    Sij_partialsky = Sij_fullsky/clc.f_sky
    Sample_covariance_full = Covariance.sample_covariance_full_sky(Z_bin, LogMass_bin, 
                                                              NHalo_bias, 
                                                              Sij_partialsky)

def lnLikelihood_theta_binned(theta):
    Om_v = theta[0]
    s8_v = theta[1]
    #ns_v = theta[2]
    #re-compute ccl cosmology
    cosmo_new = ccl.Cosmology(Omega_c = Om_v - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = s8_v, n_s=0.96)
    massdef = ccl.halos.massdef.MassDef('vir', 'critical', c_m_relation=None)
    hmd = ccl.halos.hmfunc.MassFuncDespali16(cosmo_new, mass_def=massdef)
    clc.set_cosmology(cosmo = cosmo_new, hmd = hmd, massdef = massdef)
    #re-compute integrand
    clc.compute_multiplicity_grid_MZ(z_grid = z_grid, logm_grid = logm_grid)
    N_tot = clc.Cluster_Abundance_MZ(Redshift_bin = Z_bin, Proxy_bin = LogMass_bin, method = 'simps')
    if SSC == 'noSSC':
        return lnL.lnLikelihood_Binned_Poissonian(N_tot, Nobs)
    if SSC == 'SSC':
        cov = np.diag(N_tot.flatten()) + Sample_covariance_full 
        invcov = np.linalg.inv(cov)
        return lnL.lnLikelihood_Binned_Gaussian(N_tot, Nobs, invcov)

import time
t = time.time()
true = [0.30711, .8288]
print(lnLikelihood_theta_binned(true))
tf = time.time()
print(tf-t)

#mcmc
nwalkers = 100
initial = np.random.randn(nwalkers, len(true))*.001 + np.array(np.array(true))
sampler = emcee.EnsembleSampler(nwalkers, len(true),lnLikelihood_theta_binned,)
sampler.run_mcmc(initial, 150, progress=True)
f = sampler.get_chain(discard=0, thin=1, flat=True)
np.save(wheretosave + SSC + '_' + name_analysis + '_' + 'binned_testing_likelihood_accuracy', f)