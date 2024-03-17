import unittest
import numpy as np
import pandas as pd

from fredwrapper import FetchFred
from preprocessing import preprocess_data

import switching_dynamic_factor
import kalman_filter
import hamilton_filter
import preprocessing



fred = FetchFred(api_key='d59606a150e09c54fd5158bac863da0d')

data = fred.fetch()
y = preprocess_data(data)


class testKalmanFilter(unittest.TestCase):
    def test_can_contruct(self):
        Y = np.array([])


class testSwitchingDynamicFactor(unittest.TestCase):
    def test_can_contruct(self, y):
        model = switching_dynamic_factor.SwitchingDynamicFactor(
            endog=y, k_factors=1, factor_order=2, error_order=2, M_states=2)


class testpreprocess_data(unittest.TestCase):
    def test_can_transfrom(self):
        preprocess_data(pd.DataFrame({'One': [i for i in range(1,10)], {'Two': [i for i in range(1,10)]}))

preprocessing.preprocess_data