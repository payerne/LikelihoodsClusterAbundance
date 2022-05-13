import numpy as np
import scipy
from scipy import stats
from scipy.special import erfc
from scipy.stats import poisson
from scipy.stats import multivariate_normal
from scipy.integrate import quad,simps, dblquad

class lnLikelihood():
    r"""
        compute likelihood :
            a. for the binned gaussian case
            b. for the binned poissonian case
            c. for the un-binned poissonian case
    """
    def __init__(self, theory=None):
        self.name = 'Likelihood for cluster count Cosmology'
        self.theory = theory
        
    def lnPosterior(self, Nth, Nobs, likelihood='Poissonian'):
        r"""
        compute log-likelihood
        Attributes:
        -----------
        Nth: array
            prediction of cluster abundance
        Nobs: array
            observed cluster abundance
        likelihood: str
            type of likelihood
        Returns:
        --------
        lnL: float
            log-likelihood value
        """

        if likelihood == 'Poissonian':
            return self.lnLikelihood_Binned_Poissonian(Nth, Nobs)

        if likelihood == 'Gaussian': 
            return self.lnLikelihood_Binned_Gaussian(Nth, Nobs, 
                                                     self.theory['inv_full_covariance'])

        if likelihood == 'GaussianCholesky': 
            return self.lnLikelihood_Binned_Gaussian_Cholesky(Nth, Nobs, 
                                                              self.theory['inv_L'])

        if likelihood == 'MPG': 
            return self.lnLikelihood_Binned_MPG_Block_Diagonal(Nth, Nobs, 
                                                               self.theory['Halo_bias'], 
                                                               self.theory['S_ij'].diagonal(), 
                                                               method='exact')
    
    def lnLikelihood_Binned_Poissonian(self, N_th_matrix, N_obs_matrix):
        r"""
        returns the value of the log-likelihood for Poissonian binned approach
        Attributes:
        -----------
        N_th_matrix: array
            cosmological prediction for binned cluster abundance
        N_obs_matrix:
            observed binned cluster abundance
        Returns:
        --------
        add attributes with total log-likelihood for Poissonian binned approach
        """
        mu_list = N_th_matrix.flatten()
        #rv = poisson(mu_list)
        #x = rv.pmf(N_obs_matrix.flatten())
        #return np.sum(np.log(x))
        return np.sum(N_obs_matrix.flatten() * np.log(N_th_matrix.flatten()) - N_th_matrix.flatten())

    def lnLikelihood_Binned_Gaussian(self, N_th_matrix, N_obs_matrix, inv_covariance_matrix):
        r"""
        Attributes:
        -----------
        N_th_matrix: array
            cosmological prediction for binned cluster abundance
        N_obs_matrix:
            observed binned cluster abundance
        covariance_matrix: array
            full covariance matrix for binned cluster abundance
        Returns:
        --------
        add attributes with total log-likelihood for Gaussian binned approach
        """
        delta = (N_obs_matrix - N_th_matrix).flatten()

        return -0.5*np.sum(delta*inv_covariance_matrix.dot(delta))
    
    def lnLikelihood_Binned_Gaussian_Cholesky(self, N_th_matrix, N_obs_matrix, inv_L):
        r"""
        Attributes:
        -----------
        N_th_matrix: array
            cosmological prediction for binned cluster abundance
        N_obs_matrix:
            observed binned cluster abundance
        covariance_matrix: array
            full covariance matrix for binned cluster abundance
        Returns:
        --------
        add attributes with total log-likelihood for Gaussian binned approach
        """
        delta = (N_obs_matrix - N_th_matrix).flatten()
        return -0.5*np.sum(inv_L.dot(delta)**2)
        
    def lnLikelihood_Binned_MPG_Block_Diagonal(self, N_th_matrix, N_obs_matrix, Halo_bias, S_ii, method = 'exact'):
        r"""
        Attributes:
        -----------
        N_th_matrix: array
            cosmological prediction for binned cluster abundance
        N_obs_matrix:
            observed binned cluster abundance
        Halo_bias: array
            binned halo bias
        S_ii: array
            diagonal matter amplitude fluctuation
        Returns:
        --------
        add attributes with total log-likelihood for MPG likelihood with diagonal sample variance
        """
        
        n_z_bin, n_m_bin = N_th_matrix.shape
        n = 10
        def _integrand_(dx, n_th, n_obs, hbias, S_ii): 
            rv = poisson(n_th*(1 + hbias*dx))
            return np.prod(rv.pmf(n_obs)) * multivariate_normal.pdf(dx, mean=0, cov=S_ii)
        mvp = np.zeros(n_z_bin)
        for i in range(n_z_bin):
            n_obs, n_th = N_obs_matrix[i,:], N_th_matrix[i,:]
            hbias = Halo_bias[i,:]
            if method == 'exact': 
                min_border = max(-n*np.sqrt(S_ii[i]), max(-1/hbias))
                max_border = (n+1)*np.sqrt(S_ii[i])
                res, err = quad(_integrand_, min_border, max_border,
                               epsabs=1.49e-08, epsrel=1.49e-08,
                               args = (n_th, n_obs, hbias, S_ii[i])) 
                mvp[i] = res
        return np.sum(np.log(mvp))
    
    def lnLikelihood_Binned_MPG_Block_Diagonal_individual(self, N_th, N_obs, Halo_bias, S, method = 'exact'):
        r"""
        Attributes:
        -----------
        N_th_matrix: array
            cosmological prediction for binned cluster abundance
        N_obs_matrix:
            observed binned cluster abundance
        Halo_bias: array
            binned halo bias
        S_ii: array
            diagonal matter amplitude fluctuation
        Returns:
        --------
        add attributes with total log-likelihood for MPG likelihood with diagonal sample variance
        """
        
        n = 10
        def _integrand_(dx, n_th, n_obs, hbias, S_ii): 
            rv = poisson(n_th*(1 + hbias*dx))
            return np.prod(rv.pmf(n_obs)) * multivariate_normal.pdf(dx, mean=0, cov=S_ii)
        n_obs, n_th = N_obs, N_th
        hbias = Halo_bias
        if method == 'exact': 
            min_border = max(-n*np.sqrt(S), -1/hbias)
            max_border = (n+1)*np.sqrt(S)
            res, err = quad(_integrand_, min_border, max_border,
                           epsabs=1.49e-08, epsrel=1.49e-08,
                           args = (n_th, n_obs, hbias, S)) 
            mvp = res
        return np.log(mvp)
    
    def lnLikelihood_UnBinned_Poissonian(self, Omega, dN_dzdlogMdOmega, N_tot):
        r"""
        Attributes:
        -----------
       dN_dzdlogMdOmega: array
            cosmological prediction for multiplicu-ity function
        N_tot: float
            cosmological prediction for total number of cluster
        Returns:
        --------
        add attributes with total log-likelihood for Poissonian unbinned approach
        """
        self.lnL_UnBinned_Poissonian = np.sum(np.log(Omega * dN_dzdlogMdOmega)) - N_tot
    
    def lnLikelihood_Binned_MPG_approx(self, N_th_matrix, N_obs_matrix, sample_covariance):
        r"""
        Attributes:
        -----------
        N_obs_matrix:
            observed binned cluster abundance
        N_th_matrix: array
            cosmological prediction for binned cluster abundance
        sample_covarince: array
            sample covariance matrix for binned cluster abundance
        Returns:
        --------
        add attributes with total log-likelihood MPG approximation
        """
        n = len(N_th_matrix.flatten())
        Kronoecker = np.eye(n)
        unity = np.zeros(n) + 1
        N_obs = N_obs_matrix.flatten()
        mu = N_th_matrix.flatten()
        N_obs_frac_mu = N_obs/mu
        frac_mu_x_frac_mu = np.tensordot(1./mu, 1./mu, axes=0)
        N_obs_x_1 = np.tensordot(N_obs, unity, axes=0)
        N_obs_frac_mu_x_N_obs_frac_mu = np.tensordot(N_obs_frac_mu, N_obs_frac_mu, axes=0)
        N_obs_frac_mu_x_1 = np.tensordot(N_obs_frac_mu, unity, axes=0) 
        M = sample_covariance * ( 1. 
                                  - 2. * N_obs_frac_mu_x_1
                                  + N_obs_frac_mu_x_N_obs_frac_mu
                                  - frac_mu_x_frac_mu * N_obs_x_1 * Kronoecker )
        Poisson = np.zeros([n])
        for i, mu_ in enumerate(mu):
            Poisson[i] = self.poissonian(N_obs[i], mu_)
        self.lnL_Binned_MPG_approx = np.log( (1. + .5 * np.sum(M)) * np.prod(Poisson) ) 
        
    def lnLikelihood_Binned_MPG_delta(self, N_th_matrix, N_obs_matrix, sample_covariance):
        r"""
        Attributes:
        -----------
        N_obs_matrix:
            observed binned cluster abundance
        N_th_matrix: array
            cosmological prediction for binned cluster abundance
        sample_covarince: array
            sample covariance matrix for binned cluster abundance
        Returns:
        --------
        add attributes with total log-likelihood MPG estimator
        """
        x_th_samples = np.random.multivariate_normal(N_th_matrix.flatten(), sample_covariance, size = 1000)
        #ensure positive x_th
        x_th_samples = np.where(x_th_samples >= 0, x_th_samples, 0)
        res = np.log(self.poissonian(N_obs_matrix, x_th_samples))
        self.lnL_Binned_MPG_delta = np.sum(np.mean(res, axis = 0))
        
    def lnLikelihood_Binned_MPG_diagonal(self, N_th_matrix, N_obs_matrix, sample_variance):
        r"""
        Attributes:
        -----------
        N_th_matrix: array
            cosmological prediction for binned cluster abundance
        N_obs_matrix:
            observed binned cluster abundance
        covariance_matrix: array
            sample variance for binned cluster abundance
        Returns:
        --------
        add attributes with total log-likelihood for Gaussian & Poissonian mixture binned approach
        using the diagonal sample covariance matrix
        """
        def P_MPG_delta(N_obs, mu, var_SSC):
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
            #compute normalisation of truncated Gaussian
            K = 1 - (K3*erfc(K4)/K2)*K1
            up = mu + n_sigma_delta*np.sqrt(var_SSC)
            down = mu - n_sigma_delta*np.sqrt(var_SSC)
            u_axis = u_array*(up - down) + down
            p_mvp = np.zeros( len(u_array) )
            res = self.poissonian(N_obs, u_axis) * self.Gaussian(u_axis, mu, var_SSC)
            res = np.where(u_axis >= 0, res, 0)
            return res.T/K
        
        u_array = np.linspace(0, 1, 500)
        n_sigma_delta = 3
        _integrand_ = np.zeros([len(N_obs_matrix), len(u_array)])
        L = 1
        var_list = sample_variance
        for i, nth in enumerate(N_th_matrix.flatten()):
            p = P_MPG_delta(N_obs_matrix[i], nth, var_list[i])
            #change of variable
            alpha = 2*n_sigma_delta*np.sqrt(var_list[i])/L
            _integrand_[i,:] = alpha * p
        res = simps(_integrand_, u_array)
        self.lnL_Binned_MPG_diagonal = np.sum(np.log(res))
        
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
