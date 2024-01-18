import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/PySSC/')
import PySSC
import corner
import numpy as np
import matplotlib.pyplot as plt
import scipy
import PINOCCHIO_cat
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import covariance as covar
import utils
import pandas as pd
import abundance as cl_count
from lnlikelihood import lnLikelihood
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC/SSC_contribution')
import analysis
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

where_cat = analysis.where_cat
fsky = analysis.fsky
cat=analysis.cat_number
ra, dec, redshift, Mvir = PINOCCHIO_cat.catalog(where_cat, cat, fsky, resize=analysis.resize)
print(len(ra))

code, index_analysis = sys.argv
print(sys.argv)
zmin, zmax = analysis.analysis_binned['redshift_interval'][int(index_analysis)]
logmmin, logmmax = analysis.analysis_binned['logm_interval'][int(index_analysis)]
nzbins, nmbins = analysis.analysis_binned['nzbins'][int(index_analysis)], analysis.analysis_binned['nmbins'][int(index_analysis)]

clc = cl_count.ClusterAbundance()
clc.sky_area = (fsky)*4*np.pi
clc.f_sky = clc.sky_area/(4*np.pi)
z_grid = np.linspace(zmin, zmax, 1000)
logm_grid = np.linspace(logmmin, logmmax, 1001)

z_corner = np.linspace(zmin, zmax, nzbins+1)
log10m_corner = np.linspace(logmmin, logmmax, nmbins+1)
Z_bin = [[z_corner[i], z_corner[i+1]] for i in range(len(z_corner)-1)]
print(Z_bin)
LogMass_bin = [[log10m_corner[i], log10m_corner[i+1]] for i in range(len(log10m_corner)-1)]
Nobs, a, b = np.histogram2d(redshift, np.log10(Mvir), bins = [z_corner, log10m_corner])

#choose the halo mass function and mass definition
cosmo = ccl.Cosmology(Omega_c = Omega_c_true + Omega_b_true - 0.048254, Omega_b = 0.048254, 
                          h = 0.6777, sigma8 = sigma8_true, n_s=0.96)
massdef = ccl.halos.massdef.MassDef('vir', 'critical', c_m_relation=None)
hmd = ccl.halos.hmfunc.MassFuncDespali16(cosmo, mass_def=massdef)
halobias = ccl.halos.hbias.HaloBiasTinker10(cosmo, mass_def= massdef, mass_def_strict=True)
clc.set_cosmology(cosmo = cosmo, hmd = hmd, massdef = massdef)
clc.sky_area = fsky * 4 * np.pi
clc.f_sky = clc.sky_area/(4*np.pi)

clc.compute_multiplicity_grid_MZ(z_grid = z_grid, logm_grid = logm_grid)
Abundance = clc.Cluster_Abundance_MZ(Redshift_bin = Z_bin, Proxy_bin = LogMass_bin, method = 'simps')
clc.compute_halo_bias_grid_MZ(z_grid = z_corner, logm_grid = log10m_corner, halobiais = halobias)
NHalo_bias = clc.Nhalo_bias_MZ(Redshift_bin = Z_bin, Proxy_bin = LogMass_bin, method = 'simps')
Halo_bias = NHalo_bias/Abundance
Covariance = covar.Covariance_matrix()
default_cosmo_params = {'omega_b':cosmo['Omega_b']*cosmo['h']**2, 
                                'omega_cdm':cosmo['Omega_c']*cosmo['h']**2, 
                                'H0':cosmo['h']*100, 
                                'n_s':cosmo['n_s'], 
                                'sigma8': cosmo['sigma8'],
                                'output' : 'mPk'}

#kernels
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

approx = False
if approx == True:
    approxname = 'approx'
else: approxname = 'full'

if approx == True:
    Sbb = Covariance.sample_covariance_full_sky(Z_bin, LogMass_bin, 
                                                          Halo_bias, 
                                                          Sij_partialsky)
else:
    NNSbb = Covariance.sample_covariance_full_sky(Z_bin, LogMass_bin, 
                                                          NHalo_bias, 
                                                          Sij_partialsky)
    Cov = NNSbb + np.diag(Abundance.flatten())

    inv_cov = np.linalg.inv(Cov)

lnL = lnLikelihood()

name_binning = f'{nzbins}zx{nmbins}m'

def SSC_contribution(Om_v, s8_v, approx = False):

    cosmo_new = ccl.Cosmology(Omega_c = Om_v - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = s8_v, n_s=0.96)
    
    massdef = ccl.halos.massdef.MassDef('vir', 'critical', c_m_relation=None)
    hmd = ccl.halos.hmfunc.MassFuncDespali16(cosmo_new, mass_def=massdef)
    clc.set_cosmology(cosmo = cosmo_new, hmd = hmd, massdef = massdef)
    clc.compute_multiplicity_grid_MZ(z_grid = z_grid, logm_grid = logm_grid)
    N_tot = clc.Cluster_Abundance_MZ(Redshift_bin = Z_bin, Proxy_bin = LogMass_bin, method = 'simps')
    #NNSbb_new = Covariance.sample_covariance_full_sky(Z_bin, LogMass_bin, 
          #                                                NHalo_bias, 
          #                                                Sij_partialsky)
    #Cov = NNSbb_new + np.diag(N_tot.flatten())
    #inv_cov = np.linalg.inv(Cov)
    if approx == True:
        lnPoiss =  lnL.lnLikelihood_Binned_Poissonian(N_tot, Nobs)
        SSCcont = lnL.lnLikelihood_Binned_MPG_approx(N_tot, Nobs, Sbb)
        return lnPoiss + SSCcont, lnPoiss, SSCcont
    else:
        lnGauss =  lnL.lnLikelihood_Binned_Gaussian( N_tot, Nobs, inv_cov)
        lnPoiss =  lnL.lnLikelihood_Binned_Poissonian(N_tot, Nobs)
        return lnGauss, lnPoiss, lnGauss - lnPoiss

Om = np.linspace(0.2, .4, 200)

Om_tab = True
if Om_tab == True:
    res = []
    for Om_ in Om:
        print(1)
        res.append(SSC_contribution(Om_,sigma8_true, approx = approx))
    save_pickle([Om, res], f'/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/notebooks/Unbinned_likelihood_with_SSC/SSC_contribution/SSC/Binned_cat={cat}_Nobs={np.sum(Nobs)}_nzxnm={nzbins}x{nmbins}_z[{zmin:.2f},{zmax:.2f}]_logm[{logmmin:.2f},{logmmax:.2f}]_fsky[{fsky:.4f}].pkl')
