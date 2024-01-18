import scipy
import numpy as np
from tqdm.auto import tqdm, trange
from astropy.table import Table
import utils
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad

class Proposal():
    r"""
    a class for defining the q-proposal for importance sampling
    """
    def __init__(self, mean, cov_mean, frac_Gaussian, frac_Uniform, Om_limit, s8_limit):
        r""" attributes"""
        self.mean = mean
        self.cov_mean = cov_mean
        self.Om_limit = Om_limit
        self.s8_limit = s8_limit
        self.frac_Gaussian =frac_Gaussian
        self.frac_Uniform =frac_Uniform
        name = ['gaussian', 'uniform']
        unnormed_function = [self.gaussian_unnorm, self.uniform_unnorm]
        self.norm = {n: self.norm(q) for n, q in zip(name, unnormed_function)}
        return None
    
    def norm(self,q):
        r"""find the normalisation of the proposal q"""
        def _integrand_(y,x):
            return q([x,y])
        return dblquad(_integrand_, self.Om_limit[0], self.Om_limit[1], self.s8_limit[0], self.s8_limit[1])[0]

    def uniform_unnorm(self, x):
        r"""proposal uniform"""
        Om, s8 = x
        return 1

    def gaussian_unnorm(self, x):
        r"""proposal gaussian"""
        Om, s8 = x
        pdf = multivariate_normal.pdf([Om, s8], self.mean, self.cov_mean)
        return pdf
    
    def q(self, x):
        r"""proposal gaussian + uniform"""
        return self.frac_Gaussian*self.gaussian_unnorm(x)/self.norm['gaussian'] + self.frac_Uniform*self.uniform_unnorm(x)/self.norm['uniform']
    
    def mean_s8(self, Om):
        r"""conditional mean"""
        return self.mean[1] + (self.cov_mean[0,1]/self.cov_mean[0,0])*(Om-self.mean[0])

def compute_position_from_distribution(ndim = 2, pdf = None, pdf_max = 1, N_points = 100, limits = None):
    """  
    Uses the rejection method for generating random numbers derived from an arbitrary   
    probability distribution. 

    Parameters:
    ===========
    ndim: int
        number of input dimension of the pdf
    pdf: fct
        pdf to sample
    pdf_max: float
        maximum of the multivariate pdf
    N_points: int
        number of random samples to generate
    limits:
        limits of the pdf (list of individual 2-array limits ([x1_low, x1_up]) on each parameter)
    Returns:
    ========
    random_samples: array
        random samples
    pdf_val: array
        evaluation of the pdf at the random samples
    """
    limits = np.array(limits)
    naccept, ntrial = 0, 0 
    random_samples, pdf_val = [], []
    while naccept < N_points:  
        random_sample = np.random.uniform(low=limits[:,0], high=limits[:,1], size=ndim)
        p_rand = np.random.uniform(0,pdf_max)  
        p_true = pdf(random_sample)
        if p_rand < p_true:  
            random_samples.append(random_sample), pdf_val.append(p_true)
            naccept = naccept+1  
        ntrial = ntrial+1  
    random_samples, pdf_val = np.asarray(random_samples), np.asarray(pdf_val) 
    acceptance = float(N_points/ntrial)
    print(f"acceptance = {acceptance}")
    return random_samples, pdf_val

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

        lnp=lnposterior_tab
        print(lnp)
        mask_none = lnp==None
        print(len(mask_none[mask_none==True]))
        lnw=lnp-np.log(q_val)
        lnw_norm = lnw-np.median(lnw)
        w=np.exp(lnw_norm)
        w = w/np.max(w)
        mask = np.invert(np.isnan(w))
        return w, utils.compute_mean_dispersion_from_sample(np.array(Om)[mask], 
                                                         np.array(s8)[mask], 
                                                         np.array(w)[mask])
    else: return print('not implemented yet')