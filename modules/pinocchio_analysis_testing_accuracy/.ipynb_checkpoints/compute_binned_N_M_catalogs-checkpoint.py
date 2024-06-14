import numpy as np
import sys
import pandas as pd
import glob
import time
import pickle
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from scipy import stats

def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]

where_cat = '/sps/lsst/users/cpayerne/1000xsimulations/1000_simulations/afumagalli/catalogs/plc_14/*'
cat = glob.glob(where_cat)

#where_to_save = '/pbs/throng/lsst/users/cpayerne/ClusterLikelihoods_data/pinocchio_data/'
where_to_save=f'/sps/lsst/users/cpayerne/1000xsimulations/analysis/binned_catalogs/'

# aperture
# ra_center, dec_center = 180, 60
# f_sky_default=1/4
# f_sky = f_sky_default/1
# theta_aperture = np.arccos(1-2*f_sky)*180/np.pi #deg
# cat_center_SkyCoord =SkyCoord(ra=np.array([ra_center])*u.degree, dec=np.array([dec_center])*u.degree)

#mass-redshift binning
z_min, z_max = .2, 1
logm_min, logm_max = 14.2, 14.9
n_z_bin = [4]
n_m_bin = [8]

for i in range(3):
    
    if i!=0: continue
    nz = n_z_bin[i]
    nm = n_m_bin[i]
    name_file = where_to_save + str(nz) +'x' +str(nm) + '_binned_catalogs_M_N.pkl'
    n_logm_bin = n_z_bin
    logm_corner = np.linspace(logm_min, logm_max, nm + 1)
    z_corner = np.linspace(z_min, z_max, nz + 1)
    logMass_bin = binning(logm_corner)
    Z_bin = binning(z_corner)
    Binned_cluster_abundance = []
    Binned_cluster_mass = []
    for s ,c in enumerate(cat):
        if s%100==0: print(s)
        cat_test = pd.read_csv(c ,sep=' ',skiprows=12, names=['M','z','dec','ra'])
        cat_test = Table.from_pandas(cat_test)#[idxcat_center]
        
        ra, dec, redshift, Mvir = cat_test['ra'], cat_test['dec'], cat_test['z'], cat_test['M']/0.6777
        
        N_count = stats.binned_statistic_2d(cat_test['z'], np.log10(Mvir), None, 'count', bins=[z_corner, logm_corner])
        M_halo = stats.binned_statistic_2d(cat_test['z'], np.log10(Mvir), 10**np.log10(Mvir), 'mean', bins=[z_corner, logm_corner])
        
        Binned_cluster_abundance.append(N_count.statistic)
        Binned_cluster_mass.append(M_halo.statistic)
    f = open(name_file,'wb')
    pickle.dump([cat, Z_bin, logMass_bin, Binned_cluster_abundance, Binned_cluster_mass], f)
    f.close()

