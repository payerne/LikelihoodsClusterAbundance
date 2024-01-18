import sys
import PySSC
import numpy as np
import scipy
import pyccl as ccl

sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import utils
import abundance 

class Mapping():
    def set_cosmology(self, cosmo = None):
        r"""
        Attributes:
        ----------
        cosmo : CCL cosmology object
            input cosmology from CCL
        """
        self.cosmo = cosmo
        self.massdef = ccl.halos.massdef.MassDef('vir', 'critical', c_m_relation=None)
        self.hmd = ccl.halos.hmfunc.MassFuncDespali16(self.cosmo, mass_def=self.massdef)
        self.abundance = abundance.ClusterAbundance()
        self.abundance.set_cosmology(cosmo = self.cosmo, massdef = self.massdef, hmd = self.hmd)
        self.halo_bias = ccl.halos.hbias.HaloBiasTinker10(self.cosmo, mass_def= self.massdef, mass_def_strict=True)
        
    def compute_Sij_map(self, redshift_intervals, fsky):
        r"""
        Attributes:
        ----------
        redshift_intervals : array
            list of redshift intervals
        f_sky : float
            sky fraction < 1
        Returns:
        -------
        Sij_partialsky : array
            Covariance of matter fluctuations for partial sky coverage
            (approximation taken from Lacasa et al. 2016 (https://arxiv.org/abs/1612.05958)
        """
        default_cosmo_params = {'omega_b':self.cosmo['Omega_b']*self.cosmo['h']**2, 
                                'omega_cdm':self.cosmo['Omega_c']*self.cosmo['h']**2, 
                                'H0':self.cosmo['h']*100, 
                                'n_s':self.cosmo['n_s'], 
                                'sigma8': self.cosmo['sigma8'],
                                'output' : 'mPk'}

        #kernels
        z_arr = np.linspace(np.min(redshift_intervals[0]),np.max(redshift_intervals[-1]),500)
        nbins_T   = len(redshift_intervals)
        windows_T = np.zeros((nbins_T,len(z_arr)))
        for i, z_bin in enumerate(redshift_intervals):
            Dz = z_bin[1]-z_bin[0]
            z_arr_cut = z_arr[(z_arr > z_bin[0])*(z_arr < z_bin[1])]
            for k, z in enumerate(z_arr):
                if ((z>z_bin[0]) and (z<=z_bin[1])):
                    windows_T[i,k] = 1  

        Sij_fullsky = PySSC.Sij_alt_fullsky(z_arr, windows_T, order=1, cosmo_params=default_cosmo_params, cosmo_Class=None, convention=0)
        Sij_partialsky = Sij_fullsky/fsky
        return Sij_partialsky
        
    def compute_halo_bias_map(self, z_grid, logm_grid):
        r"""
        Attributes:
        ----------
        z_grid : array
            redshift grid (size n)
        logm_grid : array
            logarithm mass grid (size m)
        Returns:
        -------
        halo_bias : array
            mapping of the halo bias over the redshift and mass two-dimensional grid (size m*n)
        """
        
        self.abundance.compute_halo_bias_grid_MZ(z_grid = z_grid, logm_grid = logm_grid, halobiais = self.halo_bias)
        return self.abundance.halo_biais
    
    def compute_dN_dlogMdzdOmega_map(self, z_grid, logm_grid, fsky):
        r"""
        Attributes:
        ----------
        z_grid : array
            redshift grid (size n)
        logm_grid : array
            logarithm mass grid (size m)
        f_sky : float
            sky fraction < 1
        Returns:
        -------
        hmf_tabulated : array
            mapping of the halo mass function (dn/dM) * partial derivative volume (dV/dzdOmega) * Omega_S over the redshift and mass two-dimensional grid (size m*n)
        """
        self.abundance.compute_multiplicity_grid_MZ(z_grid, logm_grid) 
        return self.abundance.dN_dzdlogMdOmega * fsky * 4 * np.pi
    
    def compute_bdNdm_zbins_and_dNdm_zbins(self, z_grid, logm_grid, dN_dlogMdz_map, halo_bias_map, redshift_intervals, fsky):
        r"""
        Attributes:
        ----------
        z_grid : array
            redshift grid (size n)
        logm_grid : array
            logarithm mass grid (size m)
        dN_dlogMdz_map : array
            mapping of the abundance per redshift and logarithmic mass bins over the z_grid and logm_grid
        halo_bias_map : array
            mapping of the halo bias per redshift and logarithmic mass bins over the z_grid and logm_grid
        redshift_intervals : array
            list of redshift intervals
        f_sky : float
            sky fraction < 1
        Returns:
        -------
            compute the single-variate function f_1(m) and f_2(m) defined by
            f_1(m) = \int_{z_1}^{z_2} dz b(m,z) dN(m,z)/dzdM 
            f_2(m) = \int_{z_1}^{z_2} dz dN(m,z)/dzdM 
            for the different redshift intervals
        """
        bdNdm_zbins = []
        dNdm_zbins = []
        integrand = dN_dlogMdz_map * halo_bias_map
        for redshift_range in redshift_intervals:
            mask = (z_grid > redshift_range[0])*(z_grid < redshift_range[1])
            bdNdm_zbins.append(np.trapz(integrand[:,mask], z_grid[mask]))
            dNdm_zbins.append(np.trapz(dN_dlogMdz_map[:,mask], z_grid[mask]))
        self.bdNdm_zbins = bdNdm_zbins
        self.dNdm_zbins = dNdm_zbins
        
    def compute_Nb_zbins(self, z_grid, logm_grid, dN_dlogMdz_map, halo_bias_map, redshift_intervals, fsky):
         r"""
        Attributes:
        ----------
        z_grid : array
            redshift grid (size n)
        logm_grid : array
            logarithm mass grid (size m)
        dN_dlogMdz_map : array
            mapping of the abundance per redshift and logarithmic mass over the z_grid and logm_grid
        halo_bias_map : array
            mapping of the halo bias per redshift and logarithmic mass over the z_grid and logm_grid
        redshift_intervals : array
            list of redshift intervals
        f_sky : float
            sky fraction < 1
        Returns:
        -------
            compute Nb = \int dm \int_{z_1}^{z_2} dz b(m,z) dN(m,z)/dzdM  
            for the different redshift intervals
        """
        integral = self.bdNdm_zbins
        Nb = []
        for i, redshift_range in enumerate(redshift_intervals):
                Nb.append(np.trapz(integral[i], logm_grid))
        self.Nb = np.array(Nb)
    
    def interp(self, z_grid, logm_grid, dN_dlogMdz_map, halo_bias_map, redshift_intervals, Sij, fsky):
        r"""
        Attributes:
        ----------
        z_grid : array
            redshift grid (size n)
        logm_grid : array
            logarithm mass grid (size m)
        dN_dlogMdz_map : array
            mapping of the abundance per redshift and logarithmic mass bins over the z_grid and logm_grid
        halo_bias_map : array
            mapping of the halo bias per redshift and logarithmic mass bins over the z_grid and logm_grid
        redshift_intervals : array
            list of redshift intervals
        Sij : array
            Covariance matrix of matter density fluctuations
        f_sky : float
            sky fraction < 1
        Returns:
        -------
        Interpolate pre-computed functions
        """
        Nhalo_bias_zbins = []
        halo_bias_zbins = []
        N_zbins = []
        for i, redshift_range in enumerate(redshift_intervals):
            Nhalo_bias_zbins.append(scipy.interpolate.interp1d(logm_grid, self.bdNdm_zbins[i], kind='cubic'))
            halo_bias_zbins.append(scipy.interpolate.interp1d(logm_grid, self.bdNdm_zbins[i]/self.dNdm_zbins[i], kind='cubic'))
            N_zbins.append(scipy.interpolate.interp1d(logm_grid, self.dNdm_zbins[i], kind='cubic'))
        self.Nhalo_bias_zbins = Nhalo_bias_zbins
        self.halo_bias_zbins = halo_bias_zbins
        self.N_zbins = N_zbins
        
    def compute_N_th(self, z_grid, logm_grid, dN_dlogMdz_map):
        r"""
        Attributes:
        ----------
        z_grid : array
            redshift grid (size n)
        logm_grid : array
            logarithm mass grid (size m)
        dN_dlogMdz_map : array
            mapping of the abundance per redshift and logarithmic mass bins over the z_grid and logm_grid
        Returns:
        -------
        Nth : float
            compute the predicted total number of clusters
        """
        integrand_dlogm = dN_dlogMdz_map 
        integrand_dz = np.trapz(integrand_dlogm, logm_grid, axis=0)
        return np.trapz(integrand_dz, z_grid)
        
    def N_map_interp_fct(self, z_sample, logm_sample, redshift_intervals):
        r"""
        Attributes:
        ----------
        z_sample : array
            cluster redshifts
        logm_sample : array
            cluster logarithmic masses
        dN_dlogMdz_map : array
            mapping of the abundance per redshift and logarithmic mass bins over the z_grid and logm_grid
        redshift_intervals : array
            list of redshift intervals
        Returns:
        -------
        dNdm_sample : array
            dN/dm evaluation for all clusters
        """
        
        dNdm_sample = []
        for i, redshift_range in enumerate(redshift_intervals):
            mask_redshift = (z_sample > redshift_range[0])*(z_sample < redshift_range[1])
            dNdm_sample.extend(self.N_zbins[i](logm_sample[mask_redshift]))
        
        return np.array(dNdm_sample)
    
    
    def compute_full_SSC_hybrid(self, z_grid, logm_grid, dN_dlogMdz_map, halo_bias_map, redshift_intervals, Sij, fsky, z_sample, logm_sample):
        r"""
        Attributes:
        ----------
        z_grid : array
            redshift grid (size n)
        logm_grid : array
            logarithm mass grid (size m)
        dN_dlogMdz_map : array
            mapping of the abundance per redshift and logarithmic mass bins over the z_grid and logm_grid
        halo_bias_map : array
            mapping of the halo bias per redshift and logarithmic mass bins over the z_grid and logm_grid
        redshift_intervals : array
            list of redshift intervals
        Sij : array
            Covariance matrix of matter density fluctuations
        f_sky : float
            sky fraction < 1
        z_sample : array
            cluster redshifts
        logm_sample : array
            cluster logarithmic masses
        Returns:
        -------
        lnW : float
            SSC contribution to the Poisson likelihood derived from the methodology in Takada & Spergel (2014)
        """
        deltaNb = []
        Nb2 = []
        Unity = np.eye(len(redshift_intervals))
        C = np.zeros([len(redshift_intervals), len(redshift_intervals)])
        for i, redshift_range_1 in enumerate(redshift_intervals):
            mask_redshift_1 = (z_sample > redshift_range_1[0])*(z_sample < redshift_range_1[1])
            b_sample_i = self.halo_bias_zbins[i](logm_sample[mask_redshift_1])
            Nb_obs_i = np.sum(b_sample_i)
            Nb_th_i = self.Nb[i]
            deltaNb.append(Nb_obs_i - Nb_th_i)
            Nb2_obs_i = np.sum(b_sample_i**2)
            Nb2.append(Nb2_obs_i)
            
        W = 1 + (1/2)*(np.sum(deltaNb * np.dot(Sij, deltaNb)) - np.sum(np.diag(Sij) * Nb2))
        return np.log(W)
    
    def compute_full_SSC_hybrid_garrell(self, z_grid, logm_grid, dN_dlogMdz_map, halo_bias_map, redshift_intervals, Sij, fsky, z_sample, logm_sample):
        r"""
        Attributes:
        ----------
        z_grid : array
            redshift grid (size n)
        logm_grid : array
            logarithm mass grid (size m)
        dN_dlogMdz_map : array
            mapping of the abundance per redshift and logarithmic mass bins over the z_grid and logm_grid
        halo_bias_map : array
            mapping of the halo bias per redshift and logarithmic mass bins over the z_grid and logm_grid
        redshift_intervals : array
            list of redshift intervals
        Sij : array
            Covariance matrix of matter density fluctuations
        f_sky : float
            sky fraction < 1
        z_sample : array
            cluster redshifts
        logm_sample : array
            cluster logarithmic masses
        Returns:
        -------
        lnW : float
            SSC contribution to the Poisson likelihood derived from the methodology in Garrel et al. (2022)
        """
        deltaNb = []
        Nb2 = []
        Sij_inv = np.linalg.inv(Sij)
        Unity = np.eye(len(redshift_intervals))
        C = np.zeros([len(redshift_intervals), len(redshift_intervals)])
        for i, redshift_range_1 in enumerate(redshift_intervals):
            mask_redshift_1 = (z_sample > redshift_range_1[0])*(z_sample < redshift_range_1[1])
            b_sample_i = self.halo_bias_zbins[i](logm_sample[mask_redshift_1])
            Nb_obs_i = np.sum(b_sample_i)
            Nb_th_i = self.Nb[i]
            deltaNb.append(Nb_obs_i - Nb_th_i)
            Nb2_obs_i = np.sum(b_sample_i**2)
            Nb2.append(Nb2_obs_i)
        for i, redshift_range_1 in enumerate(redshift_intervals):
            for j, redshift_range_2 in enumerate(redshift_intervals):
                C[i,j] = 0.5*(Sij_inv[i,j] + Nb2[i]*Unity[i,j])
        
        Cinv = np.linalg.inv(C)
        detSinv = np.linalg.det(Sij_inv)
        detC = np.linalg.det(C)
        deltaNb = np.array(deltaNb)
        
        return np.log(np.sqrt(detSinv/detC)) + (1/4)*np.sum(deltaNb * np.dot(Cinv, deltaNb))
        
    
#     def compute_NNSbb_thth(self, z_grid, logm_grid, dN_dlogMdz_map, halo_bias_map, redshift_intervals, Sij, fsky):

#         integral = self.bdNdm_zbins
#         Amn = np.zeros([len(redshift_intervals), len(redshift_intervals)])
#         for i, redshift_range_1 in enumerate(redshift_intervals):
#             for j, redshift_range_2 in enumerate(redshift_intervals):
#                 Amn[i,j] = Sij[i,j]*self.Nb[i]*self.Nb[j]
#         return np.sum(Amn.flatten())
    
#     def compute_NNSbb_thobs(self, z_grid, logm_grid, dN_dlogMdz_map, halo_bias_map, redshift_intervals, Sij, fsky, z_sample, logm_sample):
        
#         integral = self.bdNdm_zbins
#         Amn = np.zeros([len(redshift_intervals), len(redshift_intervals)])
#         for i, redshift_range_1 in enumerate(redshift_intervals):
#             mask_redshift_1 = (z_sample > redshift_range_1[0])*(z_sample < redshift_range_1[1])
#             b_sample_i = self.halo_bias_zbins[i](logm_sample[mask_redshift_1])
#             for j, redshift_range_2 in enumerate(redshift_intervals):
#                 mask_redshift_2 = (z_sample > redshift_range_2[0])*(z_sample < redshift_range_2[1])
#                 b_sample_j = self.halo_bias_zbins[j](logm_sample[mask_redshift_2])
#                 Amn[i,j] = Sij[i,j]*self.Nb[j]*np.sum(b_sample_i)
#         return np.sum(Amn.flatten())
    
#     def compute_NNSbb_obsobs(self, z_grid, logm_grid, dN_dlogMdz_map, halo_bias_map, redshift_intervals, Sij, fsky, z_sample, logm_sample):
        
#         integral = self.bdNdm_zbins
#         Amn = np.zeros([len(redshift_intervals), len(redshift_intervals)])
#         for i, redshift_range_1 in enumerate(redshift_intervals):
#             mask_redshift_1 = (z_sample > redshift_range_1[0])*(z_sample < redshift_range_1[1])
#             b_sample_i = self.halo_bias_zbins[i](logm_sample[mask_redshift_1])
#             for j, redshift_range_2 in enumerate(redshift_intervals):
#                 mask_redshift_2 = (z_sample > redshift_range_2[0])*(z_sample < redshift_range_2[1])
#                 b_sample_j = self.halo_bias_zbins[j](logm_sample[mask_redshift_2])
#                 Amn[i,j] = Sij[i,j]*np.sum(b_sample_j)*np.sum(b_sample_i)
#         return np.sum(Amn.flatten())
    
#     def compute_NSb2_obs(self, z_grid, logm_grid, dN_dlogMdz_map, halo_bias_map, redshift_intervals, Sij, fsky, z_sample, logm_sample):
        
#         S = []
#         for i, redshift_range_1 in enumerate(redshift_intervals):
#             mask_redshift_1 = (z_sample > redshift_range_1[0])*(z_sample < redshift_range_1[1])
#             b_sample_i = self.halo_bias_zbins[i](logm_sample[mask_redshift_1])
#             S.append(np.sum(b_sample_i**2)*Sij[i,i])
#         return np.sum(S)
            
        
                #Amn = np.zeros([len(redshift_intervals), len(redshift_intervals)])
        #for i, redshift_range_1 in enumerate(redshift_intervals):
        #    mask_redshift_1 = (z_sample > redshift_range_1[0])*(z_sample < redshift_range_1[1])
        #    b_sample_i = self.halo_bias_zbins[i](logm_sample[mask_redshift_1])
        #    Nb_obs_i = np.sum(b_sample_i)
        #    Nb_th_i = self.Nb[i]
        #    for j, redshift_range_2 in enumerate(redshift_intervals):
        #        mask_redshift_2 = (z_sample > redshift_range_2[0])*(z_sample < redshift_range_2[1])
        #        b_sample_j = self.halo_bias_zbins[j](logm_sample[mask_redshift_2])
        #        Nb_obs_j = np.sum(b_sample_j)
        #        Nb_th_j = self.Nb[j]
        #        if i==j: Delta = 1
        #        else: Delta = 0
        #        Amn[i,j] = Sij[i,j]*((Nb_th_i - Nb_obs_i)*(Nb_th_j - Nb_obs_j) - Delta * np.sum(b_sample_i * b_sample_i))
        
        #return np.log(1 + 0.5*np.sum(Amn))
        
           #lnW = 0
        #for i, redshift_range in enumerate(redshift_intervals):
        #    mask_redshift = (z_sample > redshift_range[0])*(z_sample < redshift_range[1])
        #    b_sample_i = self.halo_bias_zbins[i](logm_sample[mask_redshift])
        #    Nb_obs_i = np.sum(b_sample_i)
        #    Nb2_obs_i = np.sum(b_sample_i**2)
        #    Nb_th_i = self.Nb[i]
        #    Smm = Sij[i,i]
        #    print(max(b_sample_i), min(b_sample_i))
        #    print(Smm)
        #    Var_corr = 1 + Smm * Nb2_obs_i
         #   lnWm = (Smm/(2*Var_corr))*(Nb_th_i - Nb_obs_i)**2 - 0.5 * np.log(Var_corr)
            #lnWm = Smm*Nb_th_i**2/2 + np.log(1 - Smm*Nb_obs_i*Nb_th_i + 0.5*Smm*(Smm*Nb_th_i**2 + 1)*(Nb_obs_i**2 - Nb2_obs_i))
         #   lnW = lnW + lnWm
         #   print(lnW)
        #return lnW
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        