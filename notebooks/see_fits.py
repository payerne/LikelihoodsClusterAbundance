import sys, glob
import numpy as np
from astropy.table import Table
sys.path.append('/pbs/throng/lsst/users/cpayerne/ClusterLikelihoods/modules/')
import abundance as cl_count
import covariance as covar
import mvp_pdf
import utils
import edit

def mean_var_covar(key = None):
    r"""see data"""
    meanOm, means8, stdOm, stds8, covar= [], [], [], [], []

    files = glob.glob(key)
    for f in files:
        data_loaded = edit.load_pickle(f)
        try:
            data = Table(data_loaded)
            for i, d in enumerate(data):
                meanOm.append(d['mean_Om'])
                means8.append(d['mean_s8'])
                std_Om = d['covariance'][0,0]**.5
                std_s8 = d['covariance'][1,1]**.5
                stdOm.append(std_Om)
                stds8.append(std_s8)
                covar.append(d['covariance'])
        except:
            data = data_loaded
            meanOm.append(data['mean_Om'])
            means8.append(data['mean_s8'])
            std_Om = data['covariance'][0,0]**.5
            std_s8 = data['covariance'][1,1]**.5
            stdOm.append(std_Om)
            stds8.append(std_s8)
            covar.append(data['covariance'])
    res = Table()
    res['Om'] = np.array(meanOm)
    res['s8'] = np.array(means8)
    res['Om_std'] = np.array(stdOm)
    res['s8_std'] = np.array(stds8)
    res['cov'] = np.array(covar)
    return res

def model_tabulated(key = None):
    r"""see data"""
    t = Table(names=['Om', 's8','q','abundance'])
    files = glob.glob(key)
    for f in files:
        data_loaded = edit.load_hdf5(f)
        data_ = Table(data_loaded)
        for i, d in enumerate(data):
            data=vstack([data_, data])
    return data

        

