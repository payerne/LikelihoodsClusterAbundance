import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import fsolve
from scipy.linalg import eigh
from tqdm.auto import tqdm, trange
import multiprocessing
from matplotlib.patches import Ellipse
global func

def binning(corner): return [[corner[i],corner[i+1]] for i in range(len(corner)-1)]

def _make_ellipse(mean, cov, ax, level=0.95, color=None, label = None, ls = '-', facecolor = 'none', alpha=1):
    n_sigma = 2
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi 

    for n in np.arange(n_sigma):
        n=n+1
        ell = Ellipse(mean, 2 * n * v[0] ** 0.5, 2 * n * v[1] ** 0.5,
                                  180 + angle,
                                  linewidth=2, facecolor = facecolor,edgecolor=color, alpha=alpha,ls = ls)
        if n==1:
            ax.plot([],[],ls,color = color, label = label)
        ax.add_artist(ell)

def corner_plot_Fisher(cov_param, mean, label_param, ax, settings = None, set_axis = False):
   
    r"""
    Attributes:
    -----------
    cov_param: array
        parameter covarance of the n parameters
    mean: array
        means of parameters
    label_param: list
        name of parameters
    ax: fig
        ax
    settings: dict
        dictionnary of plot setting
    color: str
        color of contours
    set_axis: Bool
        set axis or not
    """
    plt.subplots_adjust(left=0.125,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.05, 
                            hspace=0.05)
    n_param = len(mean)
    
    #plot ellipse and gaussian single variate distribution
    for j in range(n_param):
        for i in range(n_param):
            if i == j: 
                x = np.linspace(mean[i] - 10*cov_param[i,i]**.5, mean[i] + 10*cov_param[i,i]**.5, 3000)
                ax[i,i].plot(x, multivariate_normal.pdf(x, mean=mean[i], cov=cov_param[i,i]), settings['color'])
                continue

            if j > i: continue
            index_i = i
            index_j = j
            cov_param_ij = np.zeros([2,2])
            cov_param_ij[0,0] = cov_param[j,j]
            cov_param_ij[1,1] = cov_param[i,i]
            cov_param_ij[1,0] = cov_param[i,j]
            cov_param_ij[0,1] = cov_param_ij[1,0]
            mean_ij = [mean[j], mean[i]]
            _make_ellipse(mean_ij, cov_param_ij, ax[i,j], level=0.95, color=settings['color'], 
                          label = None, ls = '-', facecolor = settings['color'], alpha=settings['alpha'])
    
    ax[0,1].plot([], [], settings['color'], label = settings['label'])
    ax[0,1].legend(fontsize=settings['legend_size'])
    if set_axis == True:
        #set xlim and ylim
        n = 3
        for i in range(n_param):
            for j in range(i+1):
            
                mean_i = mean[i]
                sigma_i = cov_param[i,i]**.5
                mean_j = mean[j]
                sigma_j = cov_param[j,j]**.5
                if j < i: 
                    ax[i,j].set_ylim(mean_i-n*sigma_i, mean_i+n*sigma_i)
                    ax[i,j].set_xlim(mean_j-n*sigma_j, mean_j+n*sigma_j)
                if j == i: 
                    ax[i,j].set_xlim(mean_j-n*sigma_j, mean_j+n*sigma_j)
                    ax[i,j].set_ylim(0)
                    
        # remove x and y ticks
        ax[0,0].set_yticks([])
        ax[0,0].set_xticks([])
        for i in range(n_param):
            for j in range(n_param):
                ax[i,j].tick_params(axis='both', which = 'major', labelsize= settings['ticks_size'])
                if j != 0: 
                    ax[i,j].set_yticks([])
                if (i > 0)*(i < n_param-1):
                     ax[i,j].set_xticks([])
        for i in range(n_param):
            for j in range(i+1,n_param):
                ax[i,j].axis('off')
                
            
        #set labl
        for j in range(n_param):
            ax[n_param-1, j].set_xlabel(label_param[j], fontsize = settings['label_size'])
        for i in range(1, n_param):
            ax[i, 0].set_ylabel(label_param[i], fontsize = settings['label_size'])   
        
                    
        return None

def _map_f(args):
    f, i = args
    return f(i)

def map(func, iter, ncores=3, ordered=True):
    
    ncpu = 3#multiprocessing.cpu_count()
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
