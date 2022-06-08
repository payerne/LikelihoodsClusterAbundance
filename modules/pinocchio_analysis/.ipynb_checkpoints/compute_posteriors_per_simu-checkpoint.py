import numpy as np
import sys, time, glob
import pyccl as ccl
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
from astropy.table import Table
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import abundance as cl_count
import importance_sampling as imp
import covariance as covar
import mvp_pdf
import edit
import utils
from lnlikelihood import lnLikelihood

name_python_file, n_z_bin, n_logm_bin, likelihood, index_simu = str(sys.argv[0]), int(sys.argv[1]), int(sys.argv[2]), str(sys.argv[3]), int(sys.argv[4])

print(sys.argv)

f_sky=1./4
f_sky_default=1./4
ratio_f_sky=f_sky/f_sky_default

#binned cluster abundance for the 1000 simualtions
dat=edit.load_pickle(f'/sps/lsst/users/cpayerne/1000xsimulations/analysis/binned_catalogs/{n_z_bin}x{n_logm_bin}_binned_catalogs.pkl')
Nobs=np.array(dat[3])[index_simu]

#Covariances
where_covar='/pbs/throng/lsst/users/cpayerne/ClusterLikelihoods_data/Covariances/'
S_ij = edit.load_pickle(where_covar+f'Sij_partialsky_blockdiagonal_{n_z_bin}x{n_logm_bin}.pickle')
full_covariance = edit.load_pickle(where_covar+f'Covariance_cluster_abudance_{n_z_bin}x{n_logm_bin}.pickle') 
#Cholesky=np.linalg.cholesky(full_covariance * ratio_f_sky)
#inv_L = np.linalg.inv(Cholesky)
inv_full_cov = np.linalg.inv(full_covariance)
Halo_bias=edit.load_pickle(where_covar+f'Halo_bias_{n_z_bin}x{n_logm_bin}.pickle') 
theory={'S_ij':(1./ratio_f_sky)*S_ij, 
        'inv_full_covariance': (1./ratio_f_sky)*inv_full_cov, 
        'Halo_bias':Halo_bias, 
        'inv_L': None}#inv_L}

#tabulated_model-in a single array
where_tab=f'/sps/lsst/users/cpayerne/1000xsimulations/analysis/'
key=f'{n_z_bin}zx{n_logm_bin}m/tabulated_model/{n_z_bin}x{n_logm_bin}_sampled_abundance'
where=where_tab+key+'*'
Nth, Om, s8, q_val=[],[],[],[]
for f in glob.glob(where):
    sampled_model=np.array(edit.load_hdf5(f))
    q=sampled_model['q']
    Nth.extend(ratio_f_sky * sampled_model['abundance'])
    q_val.extend(list(sampled_model['q']))
    Om.extend(list(sampled_model['Om']))
    s8.extend(list(sampled_model['s8']))

# mask=np.arange(10)
# Nth=np.array(Nth)[mask]
# Om=np.array(Om)[mask]
# s8=np.array(s8)[mask]
# q_val=np.array(q_val)[mask]

#likelihood
lnL=lnLikelihood(theory=theory)
def lnposterior(model, data):
    #define the posterior
    return lnL.lnPosterior(model, data, likelihood=likelihood)

t = Table()
where=f'/sps/lsst/users/cpayerne/1000xsimulations/analysis/{n_z_bin}zx{n_logm_bin}m/mean_dispersion/'
name = where+likelihood+f'_{n_z_bin}x{n_logm_bin}_index_simu_{index_simu}.pickle'
file=glob.glob(where+'*')

#if np.isin(name, file)==False:
t = {'mean_Om':None, 'mean_s8':None, 'covariance':None}
print('computing...')
mean_Om, mean_s8, covar = imp.compute_mean_covariance_importance_sampling(lnposterior, Nobs, Nth, 
                                                                 Om=Om, s8=s8, q_val=q_val, 
                                                                 mp=False, browse=False)
t['mean_Om'] = mean_Om
t['mean_s8'] = mean_s8
t['covariance'] = covar
edit.save_pickle(t, name)
