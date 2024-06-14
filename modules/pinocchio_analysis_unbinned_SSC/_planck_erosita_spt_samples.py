import sys
import corner
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.coordinates import SkyCoord
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
import pandas as pd
from astropy.table import Table
import pickle

def concatenate(n_cat, cat):
    ra, dec, redshift, Mvir = [], [], [], []
    for i in range(n_cat):
        cat_i = pd.read_csv(cat[i] ,sep=' ',skiprows=12, names=['M','z','dec','ra'])
        ra_, dec_, redshift_, Mvir_ = cat_i['ra'], cat_i['dec'], cat_i['z'], cat_i['M']/0.6777
        ra.extend(ra_)
        dec.extend(dec_)
        redshift.extend(redshift_)
        Mvir.extend(Mvir_)
    return np.array(ra), np.array(dec), np.array(redshift), np.array(Mvir)
    
def mask_fsky(fsky, cat_test, reshape=True):
    if reshape==True:
        theta_aperture = np.arccos(1-2*fsky)*180/np.pi #deg
        ra_center, dec_center = 180, 60
        cat_center_SkyCoord =SkyCoord(ra=np.array([ra_center])*u.degree, dec=np.array([dec_center])*u.degree)
        pos_ra, pos_dec = cat_test['ra'], cat_test['dec']
        cat_pinocchio_SkyCoord=SkyCoord(ra=np.array(pos_ra)*u.degree, dec=np.array(pos_dec)*u.degree)
        idxcat_center, idxcat_pinocchio, d2d, d3d = cat_center_SkyCoord.search_around_sky(cat_pinocchio_SkyCoord, theta_aperture*u.deg)
        cat_test = cat_test[idxcat_center]
    ra, dec, redshift, Mvir = cat_test['ra'], cat_test['dec'], cat_test['z'], cat_test['M']
    return ra, dec, redshift, Mvir

def spt(redshift_SPT, M500c_SPT, Mvir_SPT):
    mask_SPT_z = (redshift_SPT >= 0.25)*(redshift_SPT <= 1.5)
    mask_SPT_m = (M500c_SPT  >= 3 * 1e14)*(Mvir_SPT >= 10**(14.2))
    mask_SPT = mask_SPT_m * mask_SPT_z
    return mask_SPT

def erosita(redshift_eROSITA, M500c_eROSITA, Mvir_eROSITA):
    mask_eROSITA_m = (M500c_eROSITA  >= 2.3 * redshift_eROSITA * 1e14) 
    mask_eROSITA_m *= (M500c_eROSITA >= 7 * 1e13) * (Mvir_eROSITA >= 10**(14.2))
    mask_eROSITA_z = (redshift_eROSITA >= 0.1)*(redshift_eROSITA <= 2)
    mask_eROSITA = mask_eROSITA_m * mask_eROSITA_z
    return mask_eROSITA

def planck(redshift_Planck, M500c_Planck, Mvir_Planck):
    mask_Planck_z = (redshift_Planck >= 0.1) * (redshift_Planck <= 0.8)
    mask_Planck_m = (M500c_Planck >=  1e14*(1 + 14 * redshift_Planck ** .75))
    mask_Planck_m *= (Mvir_Planck >= 10**(14.2))
    mask_Planck = mask_Planck_m * mask_Planck_z
    return mask_Planck

def pinocchio(redshift_pinocchio, M500c_pinocchio, Mvir_pinocchio):
    mask_pinocchio_z = (redshift_pinocchio >= 0.1) * (redshift_pinocchio <= 2)
    mask_pinocchio_m = Mvir_pinocchio >= 10**(14.2)
    mask_pinocchio = mask_pinocchio_m * mask_pinocchio_z
    return mask_pinocchio

#def spt(redshift_SPT, M500c_SPT, Mvir_SPT):
#    mask_SPT_z = (redshift_SPT >= 0.25)*(redshift_SPT <= 1.2)
#    mask_SPT_m = (Mvir_SPT  >= 3*1e14)*(Mvir_SPT  <= 10**15.5)*(Mvir_SPT  >=  10**(14.2))
#    mask_SPT = mask_SPT_m * mask_SPT_z
#    return mask_SPT

#def erosita(redshift_eROSITA, M500c_eROSITA, Mvir_eROSITA):
#    mask_eROSITA_m = (Mvir_eROSITA  >= 2.3 * redshift_eROSITA * 1e14) 
    #mask_eROSITA_m *= (Mvir_eROSITA >= 7 * 1e13) * 
##    mask_eROSITA_m *=(Mvir_eROSITA >= 10**(14.2))*(Mvir_eROSITA <= 10**(15.5))
#    mask_eROSITA_z = (redshift_eROSITA >= 0.1)*(redshift_eROSITA <= 1.2)
#    mask_eROSITA = mask_eROSITA_m * mask_eROSITA_z
#    return mask_eROSITA

#def planck(redshift_Planck, M500c_Planck, Mvir_Planck):
#    mask_Planck_z = (redshift_Planck >= 0.1) * (redshift_Planck <= 0.8)
#    mask_Planck_m = (Mvir_Planck >=  1e14*(1 + 14 * redshift_Planck ** .75))
#    mask_Planck_m *= (Mvir_Planck >= 10**(14.2)) *(Mvir_Planck <= 10**15.5)
#    mask_Planck = mask_Planck_m * mask_Planck_z
#    return mask_Planck

#def pinocchio(redshift_pinocchio, M500c_pinocchio, Mvir_pinocchio):
#    mask_pinocchio_z = (redshift_pinocchio >= 0.1) * (redshift_pinocchio <= 1.2)
#    mask_pinocchio_m = Mvir_pinocchio >= 10**(14.2)
#    mask_pinocchio = mask_pinocchio_m * mask_pinocchio_z
#    return mask_pinocchio
    