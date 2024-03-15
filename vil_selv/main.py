import numpy as np
import pandas as pd

from fredwrapper import FetchFred
from switching_dynamic_factor import SwitchingDynamicFactor
from preprocessing import preprocess_data


if __name__ == "__main__":

    fred = FetchFred(api_key='d59606a150e09c54fd5158bac863da0d')

    data = fred.fetch()
    y = preprocess_data(data)

    model = SwitchingDynamicFactor(
        endog=y, k_factors=1, factor_order=2, error_order=2, M_states=2)

    result = model.fit()

    print(result.summary())
    print(y)
