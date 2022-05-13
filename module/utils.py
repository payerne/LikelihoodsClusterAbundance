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

def decorrelation_rotation_matrix_bivariate(cov):
    def f_optimize(p):
        a,b = p
        R = np.array([[a,-b],[b,a]])
        detR = np.linalg.det(R)
        R_inv = np.linalg.inv(R)
        Lambda = np.dot(R_inv, np.dot(cov_inv, R))
        res = Lambda - diag
        return res[0,0], detR - 1.
    res = fsolve(f_optimize, np.random.randn(2))
    Rotation_matrix = np.array([[res[0], -res[1]],[res[1], res[0]]])
    return Rotation_matrix

def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level

def density_contour(xdata, ydata, nbins_x, nbins_y, ax=None, c=None, **contour_kwargs):
    """ Create a density contour plot.
    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins_x : int
        Number of bins along x dimension
    nbins_y : int
        Number of bins along y dimension
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
    """
    sigma1 = 1. - np.exp(-(1./1.)**2/2.)
    sigma2 = 1. - np.exp(-(2./1.)**2/2.)
    sigma3 = 1. - np.exp(-(3./1.)**2/2.)
    H, xedges, yedges = np.histogram2d(xdata, ydata, bins=(nbins_x,nbins_y), normed=True)
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins_x))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y,1))

    pdf = (H*(x_bin_sizes*y_bin_sizes))
    pdf=gaussian_filter(pdf, sigma=.3)
    
    #one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68))
    #two_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.95))
    #three_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.99))
    one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, sigma1))
    two_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, sigma2))
    three_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, sigma3))
    levels = np.array([three_sigma, two_sigma, one_sigma])
    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T

    if ax == None:
        contour = plt.contour(X, Y, Z, levels=levels, origin="lower", colors = c, **contour_kwargs)
    else:
        contour = ax.contour(X, Y, Z, levels=levels, origin="lower", colors = c, **contour_kwargs)

    return contour

