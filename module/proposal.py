import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad

class Proposal():
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

