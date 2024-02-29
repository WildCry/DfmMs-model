import numpy as np


class HamiltonFilter:
    r'''
    See chapter 4.2.2 in Kim book
    '''

    def __init__(self) -> None:
        pass

    def predict(self):
        r'''
        Step 1:
        :math::
            Pr[S_{t}=j, S_{t-1}=i \given \psi_{t}] = Pr[S_{t}=j \given S_{t-1}=i] Pr[S_{t-1}=i\given \psi_{t-1}]
        '''
        pass

    def update(self):
        r'''
        Step 2:
        :math::
            Pr[S_{t}=j \given \psi_{t}] = \Sum_{S_{t-1}=1}^{M} Pr[S_{t}=j, S_{t-1}=i \given \psi_{t}]
        '''
        pass
