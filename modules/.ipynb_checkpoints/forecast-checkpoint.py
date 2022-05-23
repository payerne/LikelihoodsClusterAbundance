import numpy as np
import utils

class Forecast():
    r"""
    Fisher matrix:
    lnL = .5*Fisher^(-1)_{ij} dx_i dy_j + perturbative terms (non-gaussian)
    """
    def Fisher_Matrix_Gaussian(self, theta, model, cov, delta = 1e-5):
        r"""
        Attributes:
        -----------
        theta : array
            the values of model parameters used to evaluate the Fisher matrix
        model : fct
            the model
        cov : array
            the covariance matrix of the observed data
        Returns:
        --------
        Fisher_matrix : array
            the Fisher matrix for the model parameters
       """
        inv_cov=np.linalg.inv(cov)
        Fisher_matrix = np.zeros([len(theta), len(theta)])
        shape_model = cov.diagonal().shape
        fd = first_derivative(theta, model, shape_model, delta = 1e-5)
        self.fd = fd
        for i in range(len(theta)):
            for j in range(len(theta)):
                Fisher_matrix[i,j] = np.sum(fd[i]*inv_cov.dot(fd[j]))
        return Fisher_matrix
    
    def Fisher_XY(self, SigmaX, SigmaY, d_model):
        r"""
        X: true
        Y: input
        d_model: model derivatives
        
        """
        SigmaY_1 = np.linalg.inv(SigmaY)
        res=np.zeros([2,2])
        for i in range(2):
            for j in range(2):
                res[i,j] = np.sum(d_model[i,:] * np.dot(SigmaY_1, np.dot(SigmaX, SigmaY_1) ).dot(d_model[j,:]))
        return res

    def S_Fisher_Matrix(self, theta, model, cov, delta = 1e-5):

            #https://arxiv.org/pdf/1606.06455.pdf
            zeros = np.zeros(len(theta))
            first_derivative = []
            second_derivative = np.zeros([len(theta),len(theta), len(cov[:,0])])
            first_derivative = np.zeros([len(theta), len(cov[:,0])])
            def derive_i(theta, i):
                        delta_i = np.zeros(len(theta))   
                        delta_i[i] = delta
                        res = (model(np.array(theta) + delta_i/2) - \
                               model(np.array(theta) - delta_i/2))/delta
                        return res      
            for i in range(len(theta)):
                first_derivative[i] = derive_i(theta, i) 
                for j in range(len(theta)):
                    delta_j = np.zeros(len(theta))
                    delta_j[j] = delta
                    deriv_ij = (derive_i(np.array(theta) + delta_j/2, i) - \
                                derive_i(np.array(theta) - delta_j/2, i))/delta
                    second_derivative[i,j,:] = deriv_ij
            S_matrix = np.zeros([len(theta),len(theta),len(theta)])
            for k in range(len(theta)):
                for l in range(len(theta)): 
                    for m in range(len(theta)):
                        S_matrix[k,l,m] = np.sum(second_derivative[k,l]* \
                                                 np.linalg.inv(cov).dot(first_derivative[m]))
            return S_matrix

    def Q_Fisher_Matrix(self, theta, model, cov, delta = 1e-5):

       ## https://arxiv.org/pdf/1606.06455.pdf

        zeros = np.zeros(len(theta))
        second_derivative = np.zeros([len(theta),len(theta), len(cov[:,0])])
        def derive_i(theta, i):
                    delta_i = np.zeros(len(theta))
                    delta_i[i] = delta
                    res = (model(np.array(theta) + delta_i/2) - model(np.array(theta) - delta_i/2))/delta
                    return res
        for i in range(len(theta)):
            for j in range(len(theta)):
                delta_j = np.zeros(len(theta))
                delta_j[j] = delta
                deriv_ij = (derive_i(np.array(theta) + delta_j/2, i) - derive_i(np.array(theta) - delta_j/2, i))/delta
                second_derivative[i,j,:] = deriv_ij
        Q_matrix = np.zeros([len(theta),len(theta),len(theta), len(theta)])
        for k in range(len(theta)):
            for l in range(len(theta)): 
                for m in range(len(theta)):
                    for n in range(len(theta)):
                        Q_matrix[k,l,m,n] = np.sum(second_derivative[k,l]*np.linalg.inv(cov).dot(second_derivative[m,n]))
        return Q_matrix
        return lnL_Gaussian + lnL_Q_Gaussian + lnL_S_Gaussian

    def lnL_Fisher(self, theta, MLE, Fisher_matrix):
        dtheta = theta - MLE
        lnL_Gaussian = -0.5*np.sum(dtheta*Fisher_matrix.dot(dtheta))
        return lnL_Gaussian 

    def lnL_S_Fisher(self, theta, MLE, S_matrix):
        dtheta = theta - MLE
        S = 0
        for i in range(len(theta)):
            for j in range(len(theta)): 
                for k in range(len(theta)):
                    S += S_matrix[i,j,k]*dtheta[i]*dtheta[j]*dtheta[k]
        lnL_S_Gaussian = -(1./2.)*S

        return lnL_S_Gaussian

    def lnL_Q_Fisher(self, theta, MLE, Q_matrix):
        dtheta = theta - MLE
        Q = 0
        for i in range(len(theta)):
            for j in range(len(theta)): 
                for k in range(len(theta)):
                    for l in range(len(theta)):
                            Q += Q_matrix[i,j,k,l]*dtheta[i]*dtheta[j]*dtheta[k]*dtheta[l]
        lnL_Q_Gaussian = -(1./8.)*Q

        return lnL_Q_Gaussian
    
def cov_Frequentist(cov_Bayesian, d_model, Sigma):
    r"""
    compute frequentist covariance matrix
    Attributes:
    -----------
    cov_Bayesian: Bayesian covariance matrix
    d_model: 
    """
    cov_freq=np.zeros([2,2])
    for i in range(2):
        for j in range(2): 
            aT=0
            for k in range(2): aT+=cov_Bayesian[i,k]*d_model[k]
            a=0
            for k in range(2): a+=cov_Bayesian[j,k]*d_model[k]
            cov_freq[i,j] = np.sum(aT.flatten()*Sigma.dot(a.flatten()))
    return cov_freq

def first_derivative(theta, model, shape_model, delta = 1e-5):
    r"""
    ref : https://en.wikipedia.org/wiki/Finite_difference
    Attributes:
    -----------
    theta : array
        parameter values to evaluate first dreivative
    model : fct
        model to compute second derivative
    Returns:
    --------
    sec : array
        array of first derivative
    """
    first = np.zeros([len(theta)] + list(shape_model))
    for i in range(len(theta)):
        delta_i = np.zeros(len(theta))
        delta_i[i] = delta
        first[i] = (model(theta + delta_i/2) - model(theta - delta_i/2))/delta
    return first

def second_derivative(theta, model, shape_model, delta = 1e-5):
    r"""
    ref : https://en.wikipedia.org/wiki/Finite_difference
    Attributes:
    -----------
    theta : array
        parameter values to evaluate second dreivative
    model : fct
        model to compute second derivative
    Returns:
    --------
    sec : array
        array of second derivative
    """
    sec = np.zeros([len(theta),len(theta)] + list(shape_model))
    for i in range(len(theta)):
        delta_i = np.zeros(len(theta))
        delta_i[i] = delta
        for j in range(len(theta)):
            delta_j = np.zeros(len(theta))
            delta_j[j] = delta
            if i == j:
                sec[i,j] = (model(theta+delta_i) - \
                            2*model(theta) + \
                            model(theta-delta_i))/delta**2
            elif i > j:
                sec[i,j] = (model(theta+delta_i+delta_j) - \
                            model(theta+delta_i-delta_j) - \
                            model(theta-delta_i+delta_j) + \
                            model(theta-delta_i-delta_j))/(4*delta**2)
                sec[j,i] = sec[i,j]
    return sec