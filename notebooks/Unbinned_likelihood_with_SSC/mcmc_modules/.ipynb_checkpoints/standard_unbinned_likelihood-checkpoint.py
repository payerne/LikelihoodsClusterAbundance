import numpy as np

def lnLikelihood_UnBinned_Poissonian(Nth, N_tot):
       r"""
       Attributes:
       -----------
      Nth: array
           count cosmological prediction in each bin
       N_tot: float
           cosmological prediction for total number of cluster
       Returns:
       --------
       lnL : log-likelihood for Poisson statistics
       """
       return np.sum(np.log(Nth)) - N_tot
