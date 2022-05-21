import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import eigh
from tqdm.auto import tqdm, trange
import multiprocessing
from matplotlib.patches import Ellipse
global func

def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]

def _make_ellipse(mean, cov, ax, level=0.95, color=None, label = None, ls = '-'):
    n_sigma = 2
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi 

    for n in np.arange(n_sigma):
        n=n+1
        ell = Ellipse(mean, 2 * n * v[0] ** 0.5, 2 * n * v[1] ** 0.5,
                                  180 + angle,
                                  linewidth=2, facecolor = 'none',edgecolor=color, ls = ls)
        if n==1:
            ax.plot([],[],ls,color = color, label = label)
        ax.add_artist(ell)

def _map_f(args):
    f, i = args
    return f(i)

def map(func, iter, ncores=3, ordered=True):
    
    ncpu = ncores#multiprocessing.cpu_count()
    print('You have {0:1d} CPUs'.format(ncpu))
    pool = multiprocessing.Pool(processes=ncpu) 
    inputs = ((func,i) for i in iter)
    res_list = []
    if ordered: pool_map = pool.imap
    else: pool_map = pool.imap_unordered
    with tqdm(total=len(iter), desc='# progress ...') as pbar:
        for res in pool_map(_map_f, inputs):
            try :
                pbar.update()
                res_list.append(res)
            except KeyboardInterrupt:
                pool.terminate()
    pool.close()
    pool.join()
    return res_list

def weightedvar(x, w=None):
    r"""compute weighted variance"""
    if w is None :
        return np.var(x, ddof=1)
    sw = np.sum(w)
    sw2 = np.sum(w**2)
    if sw == 0.0:
        raise ZeroDivisionError
    else :
        xm = np.average(x, weights=w)
        sigma = np.average((x-xm)**2, weights=w)
        return sigma / (1. - sw2 / sw**2)

def weightedcovar(x, y, w=None):
    return np.cov(x, y, aweights=w, ddof=1)

def weightedav(x, w=None):
    r"""compute weighted average"""
    if w is None :
        return np.mean(x, ddof=1)
    else :
        return np.average(x, weights=w)

def compute_mean_dispersion_from_sample(x, y, w):
    r"""compute weighted means and covariance from 2d sample"""
    return np.array([weightedav(x, w=w), weightedav(y, w=w), weightedcovar(x, y, w=w)],dtype=object)
