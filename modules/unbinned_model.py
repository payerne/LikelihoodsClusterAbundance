import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import covariance as covar
import utils
import pandas as pd
import abundance 

sys.path.append('/pbs/throng/lsst/users/cpayerne/PySSC/')
import PySSC
import numpy as np
import scipy

import pyccl as ccl
import matplotlib.pyplot as plt

class Mapping():
    def set_cosmology(self, cosmo = None):
        r"""
        Attributes:
        ----------
        cosmo : CCL cosmology object
        mass_def: CCL object
            mass definition object of CCL
        hmd: CCL object
            halo mass distribution object from CCL
        """
        self.cosmo = cosmo
        self.massdef = ccl.halos.massdef.MassDef('vir', 'critical',)# c_m_relation=None)
        self.hmd = ccl.halos.hmfunc.MassFuncDespali16(mass_def=self.massdef)
        self.abundance = abundance.ClusterAbundance()
        self.abundance.set_cosmology(cosmo = self.cosmo, massdef = self.massdef, hmd = self.hmd)
        self.halo_bias = ccl.halos.hbias.HaloBiasTinker10(mass_def= self.massdef, mass_def_strict=True)
        
    def compute_halo_bias_map(self, z_grid, logm_grid):
        
        self.abundance.compute_halo_bias_grid_MZ(z_grid = z_grid, logm_grid = logm_grid, halobiais = self.halo_bias)
        return self.abundance.halo_biais
    
    def compute_dN_dlogMdzdOmega_map(self, z_grid, logm_grid, fsky):
        
        self.abundance.compute_multiplicity_grid_MZ(z_grid, logm_grid) 
        return self.abundance.dN_dzdlogMdOmega * fsky * 4 * np.pi
    
    def compute_sigma2_map(self, z_grid, fsky):
        
        default_cosmo_params = {'omega_b':self.cosmo['Omega_b']*self.cosmo['h']**2, 
                                'omega_cdm':self.cosmo['Omega_c']*self.cosmo['h']**2, 
                                'H0':self.cosmo['h']*100, 
                                'n_s':self.cosmo['n_s'], 
                                'sigma8': self.cosmo['sigma8'],
                                'output' : 'mPk'}
        
        return PySSC.sigma2_fullsky(z_grid, cosmo_params=default_cosmo_params, cosmo_Class=None)/fsky
    
    def compute_N_th(self, z_grid, logm_grid, dN_dlogMdz_map):
        r"""
        Compute total number of clusters
        """
        integrand_dlogm = dN_dlogMdz_map 
        integrand_dz = np.trapz(integrand_dlogm, logm_grid, axis=0)
        return np.trapz(integrand_dz, z_grid)
    
    def interp(self, z_grid, logm_grid, sigma2_map, dN_dlogMdz_map, halo_bias_map):
        r"""
        interpolate functions
        """
        self.b_map_interp_fct  = scipy.interpolate.RectBivariateSpline(logm_grid, z_grid, halo_bias_map)
        self.Nb_map_interp_fct = scipy.interpolate.RectBivariateSpline(logm_grid, z_grid, dN_dlogMdz_map * halo_bias_map)
        self.N_map_interp_fct = scipy.interpolate.RectBivariateSpline(logm_grid, z_grid, dN_dlogMdz_map)
        self.sigma2_interp_fct = scipy.interpolate.RectBivariateSpline(z_grid, z_grid, sigma2_map)
        
    def create_reduced_sample(self, z_sample, logm_sample, n_samples):

        self.n_samples_select_reduced = min(n_samples, len(z_sample))
        index = np.random.choice(np.arange(len(logm_sample)), size=self.n_samples_select_reduced, replace=False)
        self.logm_sample_select_reduced = logm_sample[index]
        self.z_sample_select_reduced = z_sample[index]
        
    def compute_NNSbb_thth(self, z_grid, logm_grid, sigma2_map, dN_dlogMdz_map, halo_bias_map, Nth_tot):
        r"""
        Compute <NNbb>_th/th
        """
        integrand_dlogm = np.multiply(dN_dlogMdz_map, halo_bias_map)
        integrand_dz = np.trapz(integrand_dlogm, logm_grid, axis = 0)
        integrand_dz_2d = np.zeros([len(z_grid), len(z_grid)])
        for i, z in enumerate(z_grid): 
            integrand_dz_2d[:,i] = integrand_dz
        NNSbb_th = np.trapz(np.trapz(sigma2_map * integrand_dz_2d, z_grid, axis = 0) * integrand_dz, z_grid)
        return NNSbb_th
    
    def compute_NNSbb_obsobs(self, z_grid, logm_grid, sigma2_map, dN_dlogMdz_map, halo_bias_map, z_sample, logm_sample, Nobs, reduced_sample=False):
        r"""
        Compute <NNbb>_obs/obs
        """
        if reduced_sample == False:
            z_sample_select = z_sample
            logm_sample_select = logm_sample
            n_samples_select = len(z_sample_select)
            
        b = self.b_map_interp_fct(logm_sample_select, z_sample_select, grid = False)
        Z1, Z2 = np.meshgrid(z_sample_select, z_sample_select)
        sigma2ij = self.sigma2_interp_fct(Z2.flatten(), Z1.flatten(), grid = False).reshape([n_samples_select, n_samples_select])
        to_sum = b * sigma2ij.dot(b)
        return np.sum(to_sum.flatten())
    
    def compute_NNSbb_obsth(self, z_grid, logm_grid, sigma2_map, dN_dlogMdz_map, halo_bias_map, 
                            z_sample, logm_sample, Nth, Nobs, reduced_sample=False):
        r"""
        Compute <NNbb>_obs/th
        """
        if reduced_sample == False:
            z_sample_select = z_sample
            logm_sample_select = logm_sample
            n_samples_select = len(z_sample_select)

        b_sample = self.b_map_interp_fct(logm_sample_select, z_sample_select, grid = False)
        Nb_th = np.multiply(dN_dlogMdz_map, halo_bias_map)
        integrand_dz_th = np.trapz(Nb_th, logm_grid, axis=0)
        
        res = np.zeros(len(z_grid))
        for i, z in enumerate(z_grid):
            sigma2_sample = self.sigma2_interp_fct(z_sample_select, np.linspace(z, z, len(z_sample_select)), grid = False)
            res[i] =  np.sum(b_sample * sigma2_sample)
        return np.trapz(integrand_dz_th * res, z_grid)

    def compute_NSb2_obs(self, z_grid, logm_grid, sigma2_map, dN_dlogMdz_map, halo_bias_map, 
                            z_sample, logm_sample, Nth, Nobs, reduced_sample=False):
        r"""
        Compute <N^2b^2>_obs
        """
        if reduced_sample == False:
            z_sample_select = z_sample
            logm_sample_select = logm_sample
            n_samples_select = len(z_sample_select)

        b_sample = self.b_map_interp_fct(logm_sample_select, z_sample_select, grid = False)
        sigma2_sample = self.sigma2_interp_fct(z_sample_select, z_sample_select, grid = False)
        return np.sum(b_sample ** 2 * sigma2_sample)
    
    def compute_W_SSC(self, NNSbb_obsobs, NNSbb_obsth, NNSbb_thth, NSb2_obs):
                    
        W = 1+0.5*(NNSbb_thth - 2*NNSbb_obsth + NNSbb_obsobs - NSb2_obs)
        if W < 0: return -np.inf
        return np.log(W)
            
            
        
                
    #def compute_sampled_maps(z_sample, logm_sample):   
    #  self.b_samples = self.b_map_interp_fct(logm_sample, z_sample, grid = False)
        