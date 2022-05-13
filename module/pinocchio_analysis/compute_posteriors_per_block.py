import numpy as np
import sys, time, glob
import pyccl as ccl
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]
from astropy.table import Table
import compute_mean_covariance_importance_sampling as Oms8
sys.path.append('/pbs/throng/lsst/users/cpayerne/ClusterLikelihoods/modules/')
import abundance as cl_count
import covariance as covar
import mvp_pdf
import edit
import utils
from lnlikelihood import lnLikelihood
import importance_sampling as imp_samp
Omega_c_true = 0.30711 - 0.048254
Omega_b_true = 0.048254
sigma8_true = .8288
Omegam_true = 0.30711

n_z_bin, n_logm_bin = int(sys.argv[1]), int(sys.argv[2])
likelihood, index_simu_min, index_simu_max = str(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

True_value = [Omega_c_true+Omega_b_true, sigma8_true]
cosmo = ccl.Cosmology(Omega_c = Omega_c_true, Omega_b = Omega_b_true, h = 0.6777, sigma8 = sigma8_true, n_s=0.96)

f_sky=1./4
f_sky_default=1./4
ratio_f_sky=f_sky/f_sky_default

#binned cluster abundance for the 1000 simualtions
dat=edit.load_pickle(f'/sps/lsst/users/cpayerne/1000xsimulations/analysis/binned_catalogs/{n_z_bin}x{n_logm_bin}_binned_catalogs.pkl')
mask=np.arange(index_simu_min, index_simu_max)
Nobs=np.array(dat[3])[mask]
n_simu=len(Nobs)

#Covariances
where_covar='/pbs/throng/lsst/users/cpayerne/ClusterLikelihoods/modules/pinocchio_analysis/Covariances/'
S_ij = edit.load_pickle(where_covar+f'Sij_partialsky_blockdiagonal_{n_z_bin}x{n_logm_bin}.pickle')
full_covariance = edit.load_pickle(where_covar+f'Covariance_cluster_abudance_full_{n_z_bin}x{n_logm_bin}.pickle') 
Cholesky=np.linalg.cholesky(full_covariance * ratio_f_sky)
inv_L = np.linalg.inv(Cholesky)
inv_full_cov = np.linalg.inv(full_covariance)
Halo_bias=edit.load_pickle(where_covar+f'Halo_bias_{n_z_bin}x{n_logm_bin}.pickle') 
theory={'S_ij':(1./ratio_f_sky)*S_ij, 
        'inv_full_covariance': (1./ratio_f_sky)*inv_full_cov, 
        'Halo_bias':Halo_bias, 
        'inv_L': inv_L}

#tabulated_model-in a single array
where_tab=f'/sps/lsst/users/cpayerne/1000xsimulations/analysis/'
key=f'{n_z_bin}zx{n_logm_bin}m/tabulated_model/{n_z_bin}x{n_logm_bin}_sampled_abundance_'
where=where_tab+key+'*'
Nth, Om, s8, q_val=[],[],[],[]
file=glob.glob(where)
for f in file:
    sampled_model=np.array(edit.load_hdf5(f))
    q=sampled_model['q']
    Nth.extend(ratio_f_sky * sampled_model['abundance'])
    q_val.extend(list(sampled_model['q']))
    Om.extend(list(sampled_model['Om']))
    s8.extend(list(sampled_model['s8']))

#likelihood
lnL=lnLikelihood(theory=theory)
def lnposterior(model, data):
    #define the posterior
    return lnL.lnPosterior(model, data, likelihood=likelihood)

t = Table()
def iter(n): return Oms8.compute_mean_covariance_importance_sampling(lnposterior, Nobs[n], Nth, 
                                                                     Om=Om, s8=s8, q_val=q_val, 
                                                                     mp=True, browse=False)
print('computing')
res = np.array([iter(n) for n in range(n_simu)])

#res = np.array(utils.map(c, np.arange(n_simu), ncores = 10, ordered=True, progress=progress))
t['mean_Om'] = res[:,0]
t['mean_s8'] = res[:,1]
t['covariance'] = res[:,2]
#where=f'/sps/lsst/users/cpayerne/1000xsimulations/analysis/{n_z_bin}zx{n_logm_bin}m/mean_dispersion/'
where='/pbs/throng/lsst/users/cpayerne/ClusterLikelihoods_data/'
name = where+likelihood +f'_{n_z_bin}x{n_logm_bin}_from_{index_simu_min}_to_{index_simu_max}.pickle'
edit.save_pickle(t, name)
