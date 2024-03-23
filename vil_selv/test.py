import unittest
import numpy as np
import pandas as pd

from fredwrapper import FetchFred
from preprocessing import preprocess_data

import switching_dynamic_factor
import kim_filter

import preprocessing



fred = FetchFred(api_key='d59606a150e09c54fd5158bac863da0d')

data = fred.fetch()
y = preprocess_data(data)


class testKimFilter(unittest,TestCase):
    def test_can_construct(self):
        pass
    
    def test_can_predict(self):
        pass
    
    def test_can_update(self):
        
    def test_can_calculate_marginal_density(self):
        pass
    
    def test_can_calculate_contitional_density(self):
        pass
    
    def test_can_smooth(self):
        pass        






# class testSwitchingDynamicFactor(unittest.TestCase):
#     def test_can_contruct(self, y):
#         model = switching_dynamic_factor.SwitchingDynamicFactor(
#             endog=y, k_factors=1, factor_order=2, error_order=2, M_states=2)


# class testpreprocess_data(unittest.TestCase):
#     def test_can_transfrom(self):
#         preprocess_data(pd.DataFrame({'One': [i for i in range(1,10)], 'Two': [i for i in range(1,10)]}))