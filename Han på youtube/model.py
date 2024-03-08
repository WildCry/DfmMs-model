import numpy as np
from  statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

class DynamicFactorModel:
    def __init__(self,endog, k_factors: int = 1, factor_AR: int = 2, error_AR: int = 2) -> None:
        
        # # letteste måten å finne  initial params er ved å bruke DFM i Statsmodels.
        # initialization_model = DynamicFactor(endog=endog, k_factors=k_factors, factor_order=factor_AR, error_order=error_AR, enforce_stationarity=False)
        # initiaalization_res = initialization_model.fit()
        # initial_params = initiaalization_res.params
        
        # self.measurenent_equation = 
        ...
        
        
        
    def predict(self, dt: float) -> None:
        # xi = alpha + T xi
        
        alpha = np.array([regime_const, 0, 0, 0, 0, 0])
        T = 