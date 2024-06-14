import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/PySSC/')
import PySSC
import corner
import numpy as np
import matplotlib.pyplot as plt
import scipy
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import covariance as covar
import utils
import pandas as pd
import abundance as cl_count
from lnlikelihood import lnLikelihood
import _analysis_mock_samples as analysis
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import PINOCCHIO_cat

def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()

import pyccl as ccl
import pickle
import edit
import h5py, glob
Omega_c_true = 0.30711 - 0.048254
Omega_b_true = 0.048254
sigma8_true = .8288
Omegam_true = 0.30711
True_value = [Omega_c_true + Omega_b_true, sigma8_true]

cosmo = ccl.Cosmology(Omega_c = Omegam_true - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = sigma8_true, n_s=0.96)


code, likelihood, index_analysis = sys.argv
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

#####
clc = cl_count.ClusterAbundance()
clc.sky_area = (fsky)*4*np.pi
clc.f_sky = clc.sky_area/(4*np.pi)
z_grid = np.linspace(zmin, zmax, 1000)
logm_grid = np.linspace(logmmin, logmmax, 1001)

nzbins = 3
nmbins = 5
z_corner = np.linspace(zmin, zmax, nzbins+1)
log10m_corner = np.linspace(logmmin, logmmax, nmbins+1)
Z_bin = [[z_corner[i], z_corner[i+1]] for i in range(len(z_corner)-1)]
LogMass_bin = [[log10m_corner[i], log10m_corner[i+1]] for i in range(len(log10m_corner)-1)]
Nobs, a, b = np.histogram2d(redshift, np.log10(Mvir), bins = [z_corner, log10m_corner])

#choose the halo mass function and mass definition
cosmo = ccl.Cosmology(Omega_c = Omega_c_true + Omega_b_true - 0.048254, Omega_b = 0.048254, 
                          h = 0.6777, sigma8 = sigma8_true, n_s=0.96)
massdef = ccl.halos.massdef.MassDef('vir', 'critical')
hmd = ccl.halos.hmfunc.MassFuncDespali16(mass_def=massdef)
halobias = ccl.halos.hbias.HaloBiasTinker10(mass_def= massdef, mass_def_strict=True)
clc.set_cosmology(cosmo = cosmo, hmd = hmd, massdef = massdef)
clc.sky_area = fsky * 4 * np.pi
clc.f_sky = clc.sky_area/(4*np.pi)

clc.compute_multiplicity_grid_MZ(z_grid = z_grid, logm_grid = logm_grid)
clc.compute_halo_bias_grid_MZ(z_grid = z_corner, logm_grid = log10m_corner, halobiais = halobias)
default_cosmo_params = {'omega_b':cosmo['Omega_b']*cosmo['h']**2, 
                                'omega_cdm':cosmo['Omega_c']*cosmo['h']**2, 
                                'H0':cosmo['h']*100, 
                                'n_s':cosmo['n_s'], 
                                'sigma8': cosmo['sigma8'],
                                'output' : 'mPk'}

z_arr = np.linspace(0.2,1.2,1000)
nbins_T   = len(Z_bin)
windows_T = np.zeros((nbins_T,len(z_arr)))
for i, z_bin in enumerate(Z_bin):
    Dz = z_bin[1]-z_bin[0]
    z_arr_cut = z_arr[(z_arr > z_bin[0])*(z_arr < z_bin[1])]
    for k, z in enumerate(z_arr):
        if ((z>z_bin[0]) and (z<=z_bin[1])):
            windows_T[i,k] = 1  

Sij_fullsky = PySSC.Sij_alt_fullsky(z_arr, windows_T, order=1, cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0)
Sij_partialsky = Sij_fullsky/clc.f_sky


lnL = lnLikelihood()

name_binning = f'{nzbins}zx{nmbins}m'

def SSC_contribution(Om_v, s8_v, approx = False):

    cosmo_new = ccl.Cosmology(Omega_c = Om_v - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = s8_v, n_s=0.96)
    
    clc.set_cosmology(cosmo = cosmo_new, hmd = hmd, massdef = massdef)
    clc.compute_multiplicity_grid_MZ(z_grid = z_grid, logm_grid = logm_grid)
    N_pred = clc.Cluster_Abundance_MZ(Redshift_bin = Z_bin, Proxy_bin = LogMass_bin, method = 'simps')
    
    NHalo_bias_pred = clc.Nhalo_bias_MZ(Redshift_bin = Z_bin, Proxy_bin = LogMass_bin, method = 'simps')
    Covariance = covar.Covariance_matrix()
    NNSbb = Covariance.sample_covariance_full_sky(Z_bin, LogMass_bin, 
                                                      NHalo_bias_pred, 
                                                      Sij_partialsky)
    
    Cov = NNSbb + np.diag(N_pred.flatten())

    inv_cov = np.linalg.inv(Cov)
    lnPoiss =  lnL.lnLikelihood_Binned_Poissonian(N_pred, Nobs)
    lnGauss =  lnL.lnLikelihood_Binned_Gaussian(N_pred, Nobs, inv_cov)
    return lnGauss, lnPoiss, 1, 1, 1, 1, 1, 1

ww = SSC_contribution(Omega_c_true + Omega_b_true, sigma8_true)
Om = np.linspace(0.2, .5, 1000)
s8 = np.linspace(0.6, 1, 35)
print(likelihood)
name_save = f'/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC/SSC_contribution/SSC_paper/{likelihood}_cat={cat}_sample_{analysis_name}.pkl'
Om_tab = True
if Om_tab == True:
    res = []
    for Om_ in Om:
        print(1)
        res.append(SSC_contribution(Om_,sigma8_true))
    save_pickle([Om, np.array(res)], name_save)
