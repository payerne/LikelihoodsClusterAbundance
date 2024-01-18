import numpy as np
import time, pickle, glob
from tqdm.auto import tqdm, trange
import pyccl as ccl
import sys
from astropy.table import Table
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import abundance as cl_count
import utils
import edit
n_z_bin, n_m_bin = int(sys.argv[1]), int(sys.argv[2])
start, end =int(sys.argv[3]), int(sys.argv[4])
def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]

param_samples = edit.load_pickle('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/pinocchio_analysis/proposal_Oms8_sup_5e14Msun.pkl')
pos = np.array([np.array(param_samples['Om']), np.array(param_samples['s8'])]).T
indexes=np.arange(start, end)
#resize positions
pos=pos[indexes]
n=len(indexes)
where=f'/sps/lsst/users/cpayerne/1000xsimulations/analysis/{n_z_bin}zx{n_m_bin}m/tabulated_model_sup_5e14Msun/'
name = where+f'{n_z_bin}x{n_m_bin}_sampled_abundance_from_{start}_to_{end}.pickle'

#z_corner = np.linspace(0.2, 1.2, n_z_bin + 1)
#logm_corner = np.linspace(14.2, 15.6, n_m_bin + 1)
z_corner = np.linspace(0.2, 1.2, n_z_bin + 1)
logm_corner = np.linspace(np.log10(5e14), 15.6, n_m_bin + 1)
Z_bin = binning(z_corner)
logMass_bin = binning(logm_corner)

z_grid = np.linspace(0.18, 1.21, 2000)
logm_grid = np.linspace(np.log10(4.9e14),15.65, 2000)

clc = cl_count.ClusterAbundance()
clc.sky_area = (0.25)*4*np.pi
clc.f_sky = clc.sky_area/(4*np.pi)
clc.z_grid=z_grid
clc.logm_grid=logm_grid

def model(theta):
    "predictiing cluster count"
    Om_v, s8_v = theta
    #re-compute ccl cosmology
    cosmo_new = ccl.Cosmology(Omega_c = Om_v - 0.048254, Omega_b = 0.048254, 
                              h = 0.6777, sigma8 = s8_v, n_s=0.96)
    massdef = ccl.halos.massdef.MassDef('vir', 'critical', c_m_relation=None)
    hmd = ccl.halos.hmfunc.MassFuncDespali16(cosmo_new, mass_def=massdef)
    clc.set_cosmology(cosmo = cosmo_new, hmd = hmd, massdef = massdef)
    #re-compute integrand
    clc.compute_multiplicity_grid_MZ(z_grid = z_grid, logm_grid = logm_grid)
    return clc.Cluster_Abundance_MZ(Redshift_bin = Z_bin, Proxy_bin = logMass_bin, 
                                    method = 'simps')

t = Table()
t['Om'], t['s8'] = pos[:,0], pos[:,1]
t['q'] = np.array(param_samples['q'][indexes])

def fct_n(n): return model(pos[n])
file=glob.glob(where+'*')
if np.isin(name, file)==False:
    print('computing')
    ti = time.time()
    #res_mp=np.array(utils.map(fct_n, np.arange(n), ncores=8, ordered=True, progress=True))
    res_mp=np.array([model(theta) for theta in pos])
    tf = time.time()
    t['abundance'] = res_mp
    edit.save_pickle(t, name)
else: print('already computed :)')
