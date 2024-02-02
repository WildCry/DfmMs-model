import numpy as np
import statsmodels as sm
from statsmodels.tools.data import _is_using_pandas


class DfmMS(sm.tsa.statespace.MLEModel):
    r'''
    Dynamic factor model with Markov switching

    Paramenters
    -----------
    endog : array_like
        The observed time-serier process : math:'y'
    k_factors : int
        the number of unobserved factors-
    factor_order : into
        The order of the vector autoregression followed by the factors.
        error_cov_type : {'scalar', 'diaglonal', 'unstructured'}, optional 
        The structure of the covariance matrix of the overvation errror term, 
        where "unstuctured" puts no restrictions on the matrix, "diagonal" 
        requires it to be any diagonal matrix (uncorrealted errors), and 
        "scalar" requires it to be a scalar times the identity matrix. Default 
        is "diagonal". 
    error_order : int, optional
        The order of thte vector autogregression followed by the observation 
        error component. Defsault is None, corresponding to white noise errors.
    error_var : bool, optional
        Whether or not to model ther errors jointly via a vector autoregression, 
        rather than as individial autoregressions. Has no effect unless 
        'errors_order' is set. Default is False
    n_states : int
        The number of hidden states.
    switching_variance : bool, optional
        Whether or not there is regime-specific heteroskedasticity, i.e.
        whether or not the error term has a switching variance. Default is
        False.

    Notes
    -----

    The model can be written as:

    .. math::
        y_it = \lambda*F_tj


    Drop enforce stationarity....
    **kwargs
        Keytword arguemtns may be used to provide default values for state space 
        matrices or for Kalman filtering optionsl See 'Representation', and 
        'KalmanFilter' for more details.
    '''

    def __init__(self, endog, k_factors, factor_order, error_order=0,
                 error_var=False, error_cov_type='diagonal', n_regimes=2,  **kwargs):

        # Factor-related properties
        self.k_factors = k_factors
        self.factor_order = factor_order

        # Error-related properties
        self.error_order = error_order
        self.error_var = error_var and error_order > 0
        self.error_cov_type = error_cov_type

        # Switching-related properties
        self.n_regimes = n_regimes  # Note! Not implemented yet

        # We need to have an array or pandas at this point
        if not _is_using_pandas(endog, None):
            endog = np.asanyarray(endog, order='C')

        # Save some useful model orders, internally used
        k_endog = endog.shape[1] if endog.ndim > 1 else 1
        self._factor_order = max(1, self.factor_order) * self.k_factors
        self._error_order = self.error_order * k_endog

        # Calculate the number of states
        k_states = self._factor_order
        k_posdef = self.k_factors
        if self.error_order > 0:
            k_states += self._error_order
            k_posdef += k_endog

        # Test for non-multivariate endog
        if k_endog < 2:
            raise ValueError('The dynamic factors model is only valid for'
                             ' multivariate time series.')

        # Test for too many factors
        if self.k_factors >= k_endog:
            raise ValueError('Number of factors must be less than the number'
                             ' of endogenous variables.')

        # Test for invalid error_cov_type
        if self.error_cov_type not in ['scalar', 'diagonal', 'unstructured']:
            raise ValueError('Invalid error covariance matrix type'
                             ' specification.')

        # By default, initialize as stationary
        kwargs.setdefault('initialization', 'stationary')

        # Initialize the state space model
        super(DfmMS, self).__init__(
            endog, exog=exog, k_states=k_states, k_posdef=k_posdef, **kwargs
        )

        # Dunno if this is necessary...
        self.ssm._time_invariant = True

        # Initialize the components
        self.parameters = {}
        self._initialize_loadings()
        self._initialize_exog()
        self._initialize_error_cov()
        self._initialize_factor_transition()
        self._initialize_error_transition()
        self.k_params = sum(self.parameters.values())

        # Since the design matrix is time-varying, it must be
        # shaped k_endog x k_states x nobs
        # Notice that exog.T is shaped k_states x nobs, so we
        # just need to add a new first axis with shape 1
        self.ssm["design"] = exog.T[np.newaxis, :, :]  # shaped 1 x 2 x nobs
        self.ssm["selection"] = np.eye(self.k_states)
        self.ssm["transition"] = np.eye(self.k_states)

        # Which parameters need to be positive?
        self.positive_parameters = slice(1, 4)

    @property
    def param_names(self):
        return ["intercept", "var.e", "var.x.coeff", "var.w.coeff"]

    @property
    def start_params(self):
        """
        Defines the starting values for the parameters
        The linear regression gives us reasonable starting values for the constant
        d and the variance of the epsilon error
        """
        exog = sm.add_constant(self.exog)
        res = sm.OLS(self.endog, exog).fit()
        params = np.r_[res.params[0], res.scale, 0.001, 0.001]
        return params

    def transform_params(self, unconstrained):
        """
        We constraint the last three parameters
        ('var.e', 'var.x.coeff', 'var.w.coeff') to be positive,
        because they are variances
        """
        constrained = unconstrained.copy()
        constrained[self.positive_parameters] = (
            constrained[self.positive_parameters] ** 2
        )
        return constrained

    def untransform_params(self, constrained):
        """
        Need to unstransform all the parameters you transformed
        in the `transform_params` function
        """
        unconstrained = constrained.copy()
        unconstrained[self.positive_parameters] = (
            unconstrained[self.positive_parameters] ** 0.5
        )
        return unconstrained

    def update(self, params, **kwargs):
        params = super(TVRegression, self).update(params, **kwargs)

        self["obs_intercept", 0, 0] = params[0]
        self["obs_cov", 0, 0] = params[1]
        self["state_cov"] = np.diag(params[2:4])
