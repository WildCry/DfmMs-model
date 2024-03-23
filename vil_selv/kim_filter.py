import numpy as np


class KalmanFilter:
    def __init__(self,
                 initial_observation_matrix: np.matrix,
                 initial_state_vector: np.array,
                 initial_intercept_vector: np.array,
                 initial_transition_matrix: np.matrix,
                 inital_covariance_matrix: np.matrix) -> None:

        self.observation_matrix = initial_observation_matrix
        self.state_vector = initial_state_vector
        self.intercept_vector = initial_intercept_vector
        self.transition_matrix = initial_transition_matrix
        self.covariance_matrix = inital_covariance_matrix

        self.state_covariance_matrix = np.eye(len(initial_state_vector))

    def predict(self) -> None:
        # xi = alpha + T xi
        # P = T P T' + sigma

        predicted_state_vector = self.intercept_vector + \
            self.transition_matrix.dot(self.state_vector)

        predicted_state_covariance_matrix = self.transition_matrix.dot(
            self.state_covariance_matrix).dot(self.transition_matrix.T) + self.covariance_matrix

        self.state_vector = predicted_state_vector
        self.state_covariance_matrix = predicted_state_covariance_matrix

    def update(self,
               observed_values: np.array,
               observed_covariance_matrix: np.array) -> None:
        # eta = Y - Z xi
        # f = Z P Z'
        # K = P Z' inv(f)
        # xi = xi + K eta
        # P = P - K Z' P

        prediction_error = observed_values - \
            self.observation_matrix.dot(self.state_vector)

        variance_prediction_error = self.observation_matrix.dot(
            self.state_covariance_matrix).dot(self.observation_matrix.T)

        kalman_gain = self.state_covariance_matrix.dot(
            self.observation_matrix.T).dot(np.linalg.inv(variance_prediction_error))

        updated_state_vector = self.state_vector + \
            kalman_gain.dot(prediction_error)

        updated_state_covariance_matrix = self.state_covariance_matrix - \
            kalman_gain.dot(self.observation_matrix.T).dot(
                self.state_covariance_matrix)

        self.state_vector = updated_state_vector
        self.state_covariance_matrix = updated_state_covariance_matrix

    def calculate_marginal_density(self) -> None:
        pass

    def calculate_conditional_density(self) -> None:
        pass
