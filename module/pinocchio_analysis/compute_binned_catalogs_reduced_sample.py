import numpy as np
import sys
import pandas as pd
import glob
import time
import pickle
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord

def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]

where_cat = '/sps/lsst/users/cpayerne/1000xsimulations/1000_simulations/afumagalli/catalogs/plc_14/*'
cat = glob.glob(where_cat)

#where_to_save = '/pbs/throng/lsst/users/cpayerne/ClusterLikelihoods_data/pinocchio_data/'
where_to_save=f'/sps/lsst/users/cpayerne/1000xsimulations/analysis/binned_catalogs/'

#aperture
ra_center, dec_center = 180, 60
f_sky_default=1/4
f_sky = f_sky_default/10
theta_aperture = np.arccos(1-2*f_sky)*180/np.pi #deg
cat_center_SkyCoord =SkyCoord(ra=np.array([ra_center])*u.degree, dec=np.array([dec_center])*u.degree)

#mass-redshift binning
z_min, z_max = .2, 1.2
logm_min, logm_max = 14.2, 15.6
n_z_bin = [4, 20, 100]
n_m_bin = [4, 30, 100]

for i in range(3):
    if i!=2: continue
    nz = n_z_bin[i]
    nm = n_m_bin[i]
    name_file = where_to_save + str(nz) +'x' +str(nm) + '_binned_catalogs_fsky_div_10.pkl'
    n_logm_bin = n_z_bin
    logm_corner = np.linspace(logm_min, logm_max, nm + 1)
    z_corner = np.linspace(z_min, z_max, nz + 1)
    logMass_bin = binning(logm_corner)
    Z_bin = binning(z_corner)
    Binned_cluster_abundance = []
    for s ,c in enumerate(cat):
        if s%100==0: print(s)
        cat_test = pd.read_csv(c ,sep=' ',skiprows=12, names=['M','z','dec','ra'])
        
        #select in aperture
        pos_ra, pos_dec = cat_test['ra'], cat_test['dec']
        cat_pinocchio_SkyCoord=SkyCoord(ra=np.array(pos_ra)*u.degree, dec=np.array(pos_dec)*u.degree)
        idxcat_center, idxcat_pinocchio, d2d, d3d = cat_center_SkyCoord.search_around_sky(cat_pinocchio_SkyCoord, theta_aperture*u.deg)
        #resize catalog
        cat_test = Table.from_pandas(cat_test)[idxcat_center]
        
        ra, dec, redshift, Mvir = cat_test['ra'], cat_test['dec'], cat_test['z'], cat_test['M']/0.6777
        N_est, mass_edges, z_edges  = np.histogram2d(cat_test['z'], np.log10(Mvir), bins=[z_corner, logm_corner])
        Binned_cluster_abundance.append(N_est)
    f = open(name_file,'wb')
    pickle.dump([cat, Z_bin, logMass_bin, Binned_cluster_abundance], f)
    f.close()

