import unittest
import numpy as np

from fredwrapper import FetchFred
from preprocessing import preprocess_data

import kalman_filter


fred = FetchFred(api_key='d59606a150e09c54fd5158bac863da0d')

data = fred.fetch()
y = preprocess_data(data)


class testKalmanFilter(unittest.TestCase):
    def test_can_contruct(self):
        Y = np.array([])
