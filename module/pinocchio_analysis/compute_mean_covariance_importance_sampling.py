import numpy as np
import sys
sys.path.append('/pbs/throng/lsst/users/cpayerne/ClusterLikelihoods/modules/')
import utils

def compute_mean_covariance_importance_sampling(lnposterior, Nobs, Nth, 
                                                Om=None, s8=None, q_val=None, 
                                                mp=False, browse=False):
    r"""
    compute posterior on importance sampling proposal
    Attributes:
    -----------
    lnposterior: fct
        posterior
    Nobs: array
        observed cluster abundance
    Om: array
        Om samples
    s8: array
        s8 samples
    q_val: array
        proposal values
    mp: Bool
        use multiprocessing or not
    browse: Bool
        use browse or not
    Returns:
    --------
    meanOm, means8, cov: float, float, array
        mean Om, mean s8, covariance of Om & s8
    """
    n_points=len(Nth)
    if browse==False:
        global func
        def func(n_iter): return lnposterior(Nth[n_iter], Nobs)
        if mp==True:
            lnposterior_tab=np.array(utils.map(func, np.arange(n_points), 
                                                  ordered=True))
        elif mp==False:
            lnposterior_tab=np.array([func(n_iter) for n_iter in range(n_points)])
        res=lnposterior_tab
        res=res-np.median(res)
        w=np.exp(res)/q_val
        w=w/np.max(w)
        mask = np.invert(np.isnan(w))
        return utils.compute_mean_dispersion_from_sample(np.array(Om)[mask], 
                                                         np.array(s8)[mask], 
                                                         np.array(w)[mask])
    else: return print('not implemented yet')

