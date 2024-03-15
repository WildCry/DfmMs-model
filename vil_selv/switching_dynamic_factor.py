import numpy as np


class Results():
    def __init__(self, estimated_parameters=dict) -> None:
        self.estimated_parameters = estimated_parameters

    def summary(self):
        pass


class SwitchingDynamicFactor:
    r'''
    Dynamic factor model with Markov switching

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    k_factors : int
        The number of unobserved factors
    factor_order : int
        The order of the vector autoregression followed by the factors.
    error_cov_type : {'scalar', 'diagonal', 'unstructured'}, optional 
        The structure of the covariance matrix of the observation error term, 
        where "unstructured" puts no restrictions on the matrix, "diagonal" 
        requires it to be any diagonal matrix (uncorrelated errors), and 
        "scalar" requires it to be a scalar times the identity matrix. Default 
        is "diagonal". 
    error_order : int, optional
        The order of the vector autoregression followed by the observation 
        error component. Default is None, corresponding to white noise errors.
    M_states : int
        The number of hidden states in factor equation.
    switching_variance : bool, optional
        Whether or not there is regime-specific heteroskedasticity, i.e.
        whether or not the error term in factor equation has a switching variance. Default is
        True.

    Notes
    -----

    The model can be written as:

    :math::
        y_{it} = \lambda_{i}(L) F_{t} + v_{it}, i = 1,\dots,n
        F_{t} = \phi(L)\mu_{S_{t}} + \gamma(L) F_{t-1} + \eta_{it}, S_{t} = 0,\dots,M
        v_{it} = D_{i}(L) v_{it-1} + \varepsilon_{it}, i = 1,\dots,n

        \phi(L)\mu_{S_{t}} = \mu_{S_{t}} - \phi_{1} \mu_{S_{t-1}} - \phi_{2} \mu_{S_{t-2}}

        \mu_{S_{t}} = \mu_{1} S_{1t} + \mu_{2} S_{2t} + \dots + \mu_{M} S_{1M}

        \sigma_{S_{t}}^{2} = \sigma_{1}^{2} S_{1t} + \sigma_{2}^{2} S_{2t} + \dots + \sigma_{M}^{2} S_{Mt}
    '''

    def __init__(self, endog, k_factors, factor_order, error_order=0,
                 M_states=2,) -> None:

        # Factors
        self.transition_matrix = np.eye(M_states)
        self.M_states = M_states
        self.k_factors = k_factors
        self.factor_order = factor_order
        self.error_order = error_order
        self.endog = endog
        self.n_endog = endog.shape[1]

    def fit(self) -> Results:
        '''
        Fits the model using the KalmanFilter, HamiltonFilter, and KimApproximations and 
        maximizes likelihood function using the BFGS algorithim. 

        Returns a  Results object with the estimated parameters.
        '''

        estimated_parameters = None  # Not implemeted yet
        result = Results(estimated_parameters)
        return result
