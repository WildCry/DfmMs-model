import numpy as np


class KalmanFilter:
    def __init__(self, initial) -> None:
        pass


class KalmanFilter_old:
    def __init__(self, initial_x: float, initial_v: float, accel_variance: float) -> None:
        # mean of state GRV
        self._x = np.array([initial_x, initial_v])
        self._accel_variance = accel_variance
        # covariance of stsate GRV

        self._P = np.eye(2)

    def predict(self, dt: float) -> None:
        # x = F  x
        # P = F P Ft + G Gt a

        F = np.array([[1, dt], [0, 1]])
        new_x = F.dot(self._x)

        G = np.array([0.5 * dt**2, dt]).reshape((2, 1))
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accel_variance

        self._P = new_P
        self._x = new_x

    def update(self, meas_value: float, meas_variance: float):
        # y = z - H x
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x + K y
        # P = (I - K H) * P

        H = np.array([1, 0]).reshape((1, 2))

        z = np.array([meas_value])
        R = np.array([meas_variance])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self._P)

        self._x = new_x
        self._P = new_P

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def pos(self) -> float:
        return self._x[0]

    @property
    def vel(self) -> float:
        return self._x[1]
