import astropy.units as u
import h5py, glob
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
import numpy as np


    
def catalog(where_cat, number, fsky, resize=False):
    cat = glob.glob(where_cat)
    cat_test = pd.read_csv(cat[number] ,sep=' ',skiprows=12, names=['M','z','dec','ra'])
    ra, dec, redshift, Mvir = cat_test['ra'], cat_test['dec'], cat_test['z'], cat_test['M']/0.6777

    if resize:
        theta_aperture = np.arccos(1-2*fsky)*180/np.pi #deg
        ra_center, dec_center = 180, 60
        cat_center_SkyCoord =SkyCoord(ra=np.array([ra_center])*u.degree, dec=np.array([dec_center])*u.degree)
        pos_ra, pos_dec = cat_test['ra'], cat_test['dec']
        cat_pinocchio_SkyCoord=SkyCoord(ra=np.array(pos_ra)*u.degree, dec=np.array(pos_dec)*u.degree)
        idxcat_center, idxcat_pinocchio, d2d, d3d = cat_center_SkyCoord.search_around_sky(cat_pinocchio_SkyCoord, theta_aperture*u.deg)
        #resize catalog
        cat_test = Table.from_pandas(cat_test)[idxcat_center]
        ra, dec, redshift, Mvir = cat_test['ra'], cat_test['dec'], cat_test['z'], cat_test['M']/0.6777
    
    return ra, dec, redshift, Mvir
    