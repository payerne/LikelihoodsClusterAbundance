import scipy
import numpy as np
from tqdm.auto import tqdm, trange
from astropy.table import Table
import utils

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

def compute_model(pos, model = None, mp = False): 
    """
    Attributes:
    ===========
    pos: array
        position of the random samples
    model: fct
        model to tabulate
    Returns:
    ========
    model_tab: array
        tabulated model
    """
    if mp == False:
        pos_array = np.array(pos).T
        m0 = model(pos_array[0])
        model_tab = np.zeros((pos_array.shape[0],) + m0.shape)
        #loop over individual values
        for index, pos in zip(np.arange(len(pos_array)), pos_array):
            model_tab[index,:] = model(pos)
        return np.array(model_tab)
    if mp == True:
        #use python multiprocessing
        global f
        def f(n): return model(pos[n])
        return utils.map(f, np.arange(len(pos)), ordered=True)

def compute_posterior(posterior = None, model_tab = None, data = None, mp = False): 
    """
    Attributes:
    ===========
    pos: array
        position of the random samples
    model_tab: fct
        tabulated model over the random samples
    posterior: fct
        posterior to be tabulated
    Returns:
    ========
    posterior_tab: array
        tabulated posterior
    """
    if mp == False:
        posterior_tab = []
        for model in model_tab:
            posterior_tab.append(posterior(model, data))
        return np.array(posterior_tab)
    if mp == True:
        global p
        iter = np.arange(len(model_tab))
        def p(n): return posterior(model_tab[n], data)
        return utils.map(p, iter, ordered=True)

def compute_mean_and_dispersion(pos, weights):
    r"""
    Attrubutes:
    -----------
    pos: array
        positions on the parameter space
    weights: array
        weights
    Returns: 
    --------
    means, dispersions: array, array
        means, and dispersions for the parameters
    """
    pos_shape = pos.shape
    n_param = pos_shape[1]
    means = np.zeros(n_param)
    dispersions = np.zeros(n_param)
    for i in range(n_param):
        x = pos[:,i]
        means[i] = utils.weightedav(pos[:,i], w=weights)
        dispersions[i] = utils.weightedvar(pos[:,i], w=weights)
    return means, dispersions

