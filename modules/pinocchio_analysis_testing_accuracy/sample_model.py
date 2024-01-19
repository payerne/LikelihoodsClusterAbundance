import numpy as np
import time, pickle
from tqdm.auto import tqdm, trange
import pyccl as ccl
from astropy.table import Table
import importance_sampling as imp_samp
import abundance as cl_count
import edit

def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]

param_samples = edit.load_pickle('/pbs/throng/lsst/users/cpayerne/ClusterLikelihoods_data/parameter_samples/proposal_Oms8.pkl')
pos = np.array([np.array(param_samples['Om']), np.array(param_samples['s8'])]).T
print(pos.shape)
n = len(pos)

clc = cl_count.ClusterAbundance()
clc.sky_area = (0.25)*4*np.pi
clc.f_sky = clc.sky_area/(4*np.pi)

n_z_bin = 4
n_logm_bin = n_z_bin
z_corner = np.linspace(0.2, 1., n_z_bin + 1)
logm_corner = np.linspace(14.2, 15.6, n_logm_bin + 1)
Z_bin = binning(z_corner)
logMass_bin = binning(logm_corner)

z_grid = np.linspace(0., 3, 900)
logm_grid = np.linspace(12,16, 900)

def model(theta):
    Om_v, s8_v = theta
    cosmo_new = ccl.Cosmology(Omega_c = Om_v - 0.048254, Omega_b = 0.048254, h = 0.677, sigma8 = s8_v, n_s=0.96)
    massdef = ccl.halos.massdef.MassDef('vir', 'critical', c_m_relation=None)
    hmd = ccl.halos.hmfunc.MassFuncDespali16(cosmo_new, mass_def=massdef)
    clc.set_cosmology(cosmo = cosmo_new, hmd = hmd, massdef = massdef)
    clc.compute_multiplicity_grid_MZ(z_grid = z_grid, logm_grid = logm_grid)
    return clc.Cluster_Abundance_MZ(Redshift_bin = Z_bin, Proxy_bin = logMass_bin, method = 'simps')

t = Table()
t['Om'], t['s8'] = pos[:,0][np.arange(n)], pos[:,1][np.arange(n)]
t['q'] = np.array(param_samples['q'])[np.arange(n)]

def fct_n(n): return model(pos[n])
ti = time.time()
res_mp = np.array(imp_samp.map(fct_n, np.arange(n), ordered=True))
tf = time.time()
t['abundance'] = res_mp
name = '/pbs/throng/lsst/users/cpayerne/ClusterLikelihoods_data/parameter_samples/4x4_sampled_abundance.pkl'
edit.save_hdf5(t, name)
