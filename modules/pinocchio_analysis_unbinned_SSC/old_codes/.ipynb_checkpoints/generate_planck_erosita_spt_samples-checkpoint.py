import sys
import corner
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import scipy
import emcee
from astropy.coordinates import SkyCoord
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
import pandas as pd
from astropy.table import Table
import pickle
import _planck_erosita_spt_samples as _samples

def save_pickle(dat, filename, **kwargs):
    file = open(filename,'wb')
    pickle.dump(dat, file)
    file.close()

import pyccl as ccl
import h5py, glob
Omega_c_true = 0.30711 - 0.048254
Omega_b_true = 0.048254
sigma8_true = .8288
Omegam_true = 0.30711
True_value = [Omega_c_true + Omega_b_true, sigma8_true]

cosmo = ccl.Cosmology(Omega_c = Omegam_true - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = sigma8_true, n_s=0.96)


where_cat = '/sps/lsst/users/cpayerne/1000xsimulations/1000_simulations/afumagalli/catalogs/plc_14/*'
where_save = '/pbs/throng/lsst/users/cpayerne/ClusterLikelihoods_data/SPT_eRosita_Planck/'
cat = glob.glob(where_cat)

mass_in = ccl.halos.massdef.MassDef('vir', 'matter')
mass_out = ccl.halos.massdef.MassDef('500', 'critical')
concentration = conc = ccl.halos.concentration.ConcentrationDuffy08(mass_def=mass_in)
mass_ranslator = ccl.halos.massdef.mass_translator(mass_in=mass_in,
                                                   mass_out=mass_out, 
                                                   concentration=concentration)

#SPT
n_cat = 1
ra, dec, redshift, Mvir = _samples.concatenate(n_cat, cat)
cat_SPT = Table()
cat_SPT['ra'] = ra
cat_SPT['dec'] = dec
cat_SPT['z'] = redshift
cat_SPT['M'] = Mvir
Omega_SPT = 2500
Omega_full = 360**2/np.pi
ra_SPT, dec_SPT, redshift_SPT, Mvir_SPT = _samples.mask_fsky(Omega_SPT/Omega_full, cat_SPT)
M500c_SPT = np.array([mass_ranslator(cosmo, Mvir_SPT[i], 1/(1+redshift_SPT[i])) for i in range(len(Mvir_SPT))])
mask_SPT = _samples.spt(redshift_SPT, M500c_SPT, Mvir_SPT)
cat_SPT_final = Table()
cat_SPT_final['ra'] = ra_SPT[mask_SPT]
cat_SPT_final['dec'] = dec_SPT[mask_SPT]
cat_SPT_final['z'] = redshift_SPT[mask_SPT]
cat_SPT_final['M'] = Mvir_SPT[mask_SPT]
save_pickle(cat_SPT_final, where_save + 'spt.pkl')

#eROSITA
n_cat = 4
ra, dec, redshift, Mvir = _samples.concatenate(n_cat, cat)
cat_eROSITA = Table()
cat_eROSITA['ra'] = ra
cat_eROSITA['dec'] = dec
cat_eROSITA['z'] = redshift
cat_eROSITA['M'] = Mvir
Omega_eROSITA = 42100
ra_eROSITA, dec_eROSITA, redshift_eROSITA, Mvir_eROSITA = _samples.mask_fsky(Omega_eROSITA/42100, cat_eROSITA, reshape=False)
M500c_eROSITA = np.array([mass_ranslator(cosmo, Mvir_eROSITA[i], 1/(1+redshift_eROSITA[i])) for i in range(len(Mvir_eROSITA))])
mask_eROSITA = _samples.erosita(redshift_eROSITA, M500c_eROSITA, Mvir_eROSITA)
cat_eROSITA_final = Table()
cat_eROSITA_final['ra'] = ra_eROSITA[mask_eROSITA]
cat_eROSITA_final['dec'] = dec_eROSITA[mask_eROSITA]
cat_eROSITA_final['z'] = redshift_eROSITA[mask_eROSITA]
cat_eROSITA_final['M'] = Mvir_eROSITA[mask_eROSITA]
save_pickle(cat_eROSITA_final, where_save + 'erosita.pkl')

#Planck
n_cat = 2
ra, dec, redshift, Mvir = _samples.concatenate(n_cat, cat)
cat_Planck = Table()
cat_Planck['ra'] = ra
cat_Planck['dec'] = dec
cat_Planck['z'] = redshift
cat_Planck['M'] = Mvir
Omega_Planck = 42100
ra_Planck, dec_Planck, redshift_Planck, Mvir_Planck = _samples.mask_fsky(Omega_Planck/42100, cat_Planck, reshape=False)
M500c_Planck = np.array([mass_ranslator(cosmo, Mvir_Planck[i], 1/(1+redshift_Planck[i])) for i in range(len(Mvir_Planck))])
mask_Planck = _samples.planck(redshift_Planck, M500c_Planck, Mvir_Planck)
cat_Planck_final = Table()
cat_Planck_final['ra'] = ra_Planck[mask_Planck]
cat_Planck_final['dec'] = dec_Planck[mask_Planck]
cat_Planck_final['z'] = redshift_Planck[mask_Planck]
cat_Planck_final['M'] = Mvir_Planck[mask_Planck]
save_pickle(cat_Planck_final, where_save + 'planck.pkl')

#PINOCHIO
n_cat = 1
ra, dec, redshift, Mvir = _samples.concatenate(n_cat, cat)
cat_pinocchio = Table()
cat_pinocchio['ra'] = ra
cat_pinocchio['dec'] = dec
cat_pinocchio['z'] = redshift
cat_pinocchio['M'] = Mvir
mask_pinocchio = _samples.pinocchio(redshift, None, Mvir)
cat_pinocchio_final = Table()
cat_pinocchio_final['ra'] = ra[mask_pinocchio]
cat_pinocchio_final['dec'] = dec[mask_pinocchio]
cat_pinocchio_final['z'] = redshift[mask_pinocchio]
cat_pinocchio_final['M'] = Mvir[mask_pinocchio]
save_pickle(cat_pinocchio_final, where_save + 'pinocchio.pkl')

plt.figure(figsize=(14,2))
bins = [np.linspace(0, 2, 30), np.linspace(14, 15.5, 30)]
plt.subplot(131)

plt.title(f'Planck sample N = {np.sum(mask_Planck):.0f} clusters')
plt.hist2d(redshift, np.log10(Mvir),alpha=1, bins=bins,cmap='twilight', cmin=1)
plt.hist2d(redshift_Planck[mask_Planck], np.log10(Mvir_Planck[mask_Planck]),alpha=1, cmap = 'gist_rainbow',bins=bins, cmin=1 )
plt.colorbar()
plt.grid()
plt.xlim(0, 2)
plt.ylim(14., 15.5)
plt.xlabel('redshift')

plt.subplot(132)

plt.title(f'SPT sample N = {np.sum(mask_SPT):.0f} clusters')
plt.hist2d(redshift, np.log10(Mvir),alpha=1,bins=bins, cmap='twilight', cmin=1)
plt.hist2d(redshift_SPT[mask_SPT], np.log10(Mvir_SPT[mask_SPT]), cmap = 'gist_rainbow', bins=bins, cmin=1)
plt.colorbar()
plt.grid()
plt.xlim(0, 2)
plt.ylim(14., 15.5)
plt.xlabel('redshift')
plt.ylabel(r'$\log_{10}(M_{\rm vir})$')

plt.subplot(133)

plt.title(f'eROSITA sample N = {np.sum(mask_eROSITA):.1e} clusters')
plt.hist2d(redshift, np.log10(Mvir),alpha=1, bins=bins,cmap='twilight', cmin=1)
plt.hist2d(redshift_eROSITA[mask_eROSITA], np.log10(Mvir_eROSITA[mask_eROSITA]), cmap = 'gist_rainbow',bins=bins, cmin=1)
plt.colorbar()
plt.grid()
plt.xlim(0, 2)
plt.ylim(14., 15.5)
plt.xlabel('redshift')