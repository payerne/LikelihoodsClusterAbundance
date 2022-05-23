import numpy as np
import scipy
from scipy import stats
from scipy.integrate import quad,simps, dblquad
from scipy.stats import poisson
from scipy.stats import multivariate_normal
from scipy.special import erfc
import matplotlib.pyplot as plt
from scipy import interpolate
from math import factorial, gamma
import time

class MVP():
    
    def __init__(self, mu, var_SSC):
        self.mu = mu
        self.var_SSC = var_SSC
        self.array = np.linspace(0, 1, 500)
        self.n_sigma_delta = 3
    
    def Gaussian(self, x, mu, var_SSC):
        r"""
        Attributes:
        -----------
        x: array
            variable along the x axis
        mu: float
            mean of the Gaussian distribution
        var_SSC: float
            variance of the Gaussian distrubution
        Returns:
        --------
        g: array
            Gausian probability density function
        """
        return np.exp(-.5*(x-mu)**2/var_SSC)/np.sqrt(2*np.pi*var_SSC)
        
    def poissonian(self, n, mu):
        r"""
        Attributes:
        -----------
        n: array
            variable along the n axis
        mu: float
            mean of the Poisson distribution
        Returns:
        --------
        p: array
            Poisson probability function
        """
        rv = poisson(mu)
        return rv.pmf(n)
        
    def __integrand__(self, x, n, mu, var_SSC):
        r"""
        Attributes:
        -----------
        x: array
            variable along the integrand axis
        n: int
            observed cluster count
        mu: float
            cosmological prediction for cluster count
        var_SSC: float
            variance of the gaussian
        Returns:
        --------
        integrand: array
        """
        return self.poissonian(n, x) * self.Gaussian(x, mu, var_SSC)
    
    def p_mvp_delta(self, N_array, mu, var_SSC):
        r"""
        Attributes:
        -----------
        N_array: array
            cluster count axis (int values)
        mu: float
            cluster abuncance cosmological prediction
        var_SSC: folat
            SSC variance of cluster count
        Returns:
        --------
        p_mvp: array
            Gaussian/Poisson mixture probability along the overdensity axis
        r"""
        K1 = (1./np.sqrt(2.*np.pi*var_SSC))
        K2 = np.sqrt(1./var_SSC)
        K3 = np.sqrt(np.pi/2.)
        K4 = mu*K2/np.sqrt(2.)
        K = 1 - (K3*erfc(K4)/K2)*K1
        up = mu + self.n_sigma_delta*np.sqrt(var_SSC)
        down = mu - self.n_sigma_delta*np.sqrt(var_SSC)
        u_axis = self.array*(up - down) + down
        p_mvp = np.zeros([len(N_array), len(self.array)])
        def _integrand_(n, u_axis):
            return self.__integrand__(u_axis, n, mu, var_SSC)
        N_mesh, u_mesh = np.meshgrid(N_array, u_axis)
        res = _integrand_(N_mesh, u_mesh)
        res = np.where(u_mesh >= 0, res, 0)
        return res.T/K
    
    def _set_axis(self, n_sigma_N, mu_list, var_SSC_list):
        r"""
        Attributes:
        -----------
        n_sigma_N: int
            sigma
        mu_list: list
            list of mean
        var_SSC_list: list
            list of SSC variance
        Returns:
        --------
        set axis range
        """
        N_array_list = []
        split_indexes = []
        n_max = round(max(mu_list  + n_sigma_N*np.sqrt(mu_list + var_SSC_list)))
        for i, nth in enumerate(mu_list):
            min_ = max(0, round(nth - n_sigma_N*np.sqrt(nth + var_SSC_list[i])))
            max_ = min(n_max - 1, round(nth + n_sigma_N*np.sqrt(nth + var_SSC_list[i])) + 5)
            N_array = np.arange(min_, max_)
            N_array_list.append(N_array)
        self.N_array_list = N_array_list
        self.array_size = np.array([len(n_array) for n_array in N_array_list])
        self.len_n_array = np.sum(self.array_size)
        for i, nth in enumerate(mu_list):
            index0 = np.sum(np.array(self.array_size)[np.arange(i)])
            index1 = index0 + np.array(self.array_size)[i]
            indexes = np.arange(index0, index1)
            split_indexes.append(indexes)
        self.split_indexes = split_indexes
        self.indexes = np.array([s[0] for s in self.split_indexes])
        self.n_max = n_max
        
    def p_mvp(self, mu_list, var_SSC_list):
        r"""
        Attributes:
        -----------
        mu_list: array
            list of cosmological prediction
        var_SSC_list: array
            list of variance
        Returns:
        --------
        N: array
            list of count axis
        P_MVP: array
            list of MVP probability distribution
        r"""
        _integrand_ = np.zeros([self.len_n_array, len(self.array)])
        l = self.array[-1] - self.array[0]
        split_indexes = []
        for i, nth in enumerate(mu_list):
            indexes = self.split_indexes[i]
            p = self.p_mvp_delta(self.N_array_list[i], nth, var_SSC_list[i])
            alpha = 2*self.n_sigma_delta*np.sqrt(var_SSC_list[i])/l #change of variable
            _integrand_[indexes,:] = alpha * p
        P_MVP_table = simps(_integrand_, self.array)
        P_MVP = np.array([P_MVP_table[indexes] for indexes in self.split_indexes])
        return self.N_array_list, P_MVP
    
    def p_mvp_delta_obs(self, N_obs, mu, var_SSC):
        r"""
        Attributes:
        -----------
        N_array: array
            cluster count axis (int values)
        mu: float
            cluster abuncance cosmological prediction
        var_SSC: folat
            SSC variance of cluster count
        Returns:
        --------
        p_mvp: array
            Gaussian/Poisson mixture probability along the overdensity axis
        r"""
        K1 = (1./np.sqrt(2.*np.pi*var_SSC))
        K2 = np.sqrt(1./var_SSC)
        K3 = np.sqrt(np.pi/2.)
        K4 = mu*K2/np.sqrt(2.)
        K = 1 - (K3*erfc(K4)/K2)*K1
        up = mu + self.n_sigma_delta*np.sqrt(var_SSC)
        down = mu - self.n_sigma_delta*np.sqrt(var_SSC)
        u_axis = self.array*(up - down) + down
        p_mvp = np.zeros( len(self.array) )
        def _integrand_(u_axis):
            return self.__integrand__(u_axis, N_obs, mu, var_SSC)
        res = _integrand_(u_axis)
        res = np.where(u_axis >= 0, res, 0)
        return res.T/K
    
    def p_mvp_obs(self, N_obs_list, mu_list, var_SSC_list):
        r"""
        Attributes:
        -----------
        mu_list: array
            list of cosmological prediction
        var_SSC_list: array
            list of variance
        Returns:
        --------
        N: array
            list of count axis
        P_MVP: array
            list of MVP probability distribution
        r"""
        _integrand_ = np.zeros([len(N_obs_list), len(self.array)])
        l = self.array[-1] - self.array[0]
        split_indexes = []
        for i, nth in enumerate(mu_list):
            p = self.p_mvp_delta_obs(N_obs_list[i], nth, var_SSC_list[i])
            alpha = 2*self.n_sigma_delta*np.sqrt(var_SSC_list[i])/l #change of variable
            _integrand_[i,:] = alpha * p
        P_MVP_table = simps(_integrand_, self.array)
        return P_MVP_table
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            #works well
        #for i, n in enumerate(N_array):
        #    res = self.__integrand__(u_axis, n, mu, var_SSC)
        #    res = np.where(u_axis >= 0, res, 0)
        #    p_mvp[i,:] = res
        #return p_mvp/K
    
    r"""    
    def P_MVP(N_array = 1, mu = 1, var = 1, method = 'simps'):

        K1 = (1./np.sqrt(2.*np.pi*var))
        K2 = np.sqrt(1./var)
        K3 = np.sqrt(np.pi/2.)
        K4 = mu*K2/np.sqrt(2.)
        K = 1 - (K3*erfc(K4)/K2)*K1
        def poissonian(x = 1, mu = 1):
            rv = poisson(mu)
            return rv.pmf(x)
        def Gaussian(x = 1, mu = 1, var = 1):
            return multivariate_normal.pdf(x, mean=mu, cov=var)
        def __integrand__(x, n, var):
            return poissonian(x = n, mu = x) * Gaussian(x = x, mu = mu, var = var)
        P_MVP = []
        n_sigma = 3
        u_axis = np.sort(np.random.randn(500)*n_sigma*np.sqrt(var) + mu)
        u_axis = np.linspace(mu - n_sigma*np.sqrt(var), mu + n_sigma*np.sqrt(var), 500)
        u_axis = u_axis[(u_axis > mu - n_sigma*np.sqrt(var))*(u_axis < mu + n_sigma*np.sqrt(var))]
        u_axis[-1] = mu + n_sigma*np.sqrt(var)
        u_axis[0] = mu - n_sigma*np.sqrt(var)
        u_axis = u_axis[u_axis >= 0]

        #if np.sqrt(var)/mu < 1e-3:
        #    rv = poisson(mu)
        #    return rv.pmf(N_array)

        if method == 'simps':
            p_mvp = np.zeros([len(N_array), len(u_axis)])
            for i, n in enumerate(N_array):
                p_mvp[i,:] = __integrand__(u_axis, n, var)
            P_MVP = simps(p_mvp, x=u_axis, axis = 1)
            return np.array(P_MVP)/K

        if method == 'quad_interp':
            for n in N_array:
                y_array = __integrand__(u_axis, n, var)
                def __integrand_interp(x):
                    return np.interp(x, u_axis, y_array)
                p_mvp = quad(__integrand_interp, u_axis[0], u_axis[-1])[0]
                P_MVP.append(p_mvp)
            return np.array(P_MVP)/K 

        if method == 'brut_force':
            for n in N_array:
                p_mvp = quad(__integrand__, mu - 5*np.sqrt(var),
                             mu + 5*np.sqrt(var), args=(n, var))[0]
                P_MVP.append(p_mvp)
            return np.array(P_MVP)/K

        if method == 'special_function':
            B = 1. - mu/var
            C = 1./(2*var)
            P_MVP = []
            for n in N_array:
                try: 
                    factorial_ = factorial(n)
                except: factorial_ =  np.sqrt(2*np.pi)*(n/np.exp(1))**n
                coeff = K1*np.exp(-mu**2/(2*var))/factorial_
                special_integral = integral_special(n,B,C)
                p_mvp = K**(-1)*coeff*special_integral
                P_MVP.append(p_mvp)
            return np.array(P_MVP)

    def P_MVP_check(N_array = 1, mu = 1, var = 1):

        try: P_MVP(N_array = N_array, mu = mu, var = var, method = 'special_function')
        except : P_MVP(N_array = N_array, mu = mu, var = var, method = 'simps')


    def integral_special(a,b,c):
        A = 0.5*c**(-1-a/2)
        B = np.sqrt(c)*gamma(0.5+a/2)*scipy.special.hyp1f1((a+1)/2,0.5,b**2/(4*c))
        C = b*gamma(1 + a/2)*scipy.special.hyp1f1(1+a/2.,3./2.,b**2./(4*c))
        return A*(B-C)


    def P_cluster_count(N_array = 1, mu = 1, cov = 1):
        res = []
        for n in N_array:
            res.append(P_cluster_count_u(N = n, mu = mu, cov = cov))
        return np.array(res)


    def convolution_Gaussian_Poissonian_u(x = 1, mu_gauss = 1, sigma_gauss = 1, Lambda_poiss = 1):

        rv = poisson(Lambda_poiss)
        n_max = mu_gauss + sigma_gauss * 10 + 10 * Lambda_poiss + Lambda_poiss
        n = np.arange(n_max)
        Poiss = rv.pmf(n)
        Gauss = multivariate_normal.pdf(x - n, mean=mu_gauss, cov=sigma_gauss**2) 
        return np.sum(Gauss*np.array(Poiss))

    def convolution_Gaussian_Poissonian(x_array = 1, mu_gauss = 1, sigma_gauss = 1, Lambda_poiss = 1):

        res = []
        for X in x_array:
            a = convolution_Gaussian_Poissonian_u(x = X, mu_gauss = mu_gauss, sigma_gauss = sigma_gauss, Lambda_poiss = Lambda_poiss)
            res.append(a)
        return np.array(res)
    r"""
