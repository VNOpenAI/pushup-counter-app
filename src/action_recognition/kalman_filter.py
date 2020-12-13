class SimpleKalmanFilter():

    def __init__(self, mea_e,  est_e,  q, current_estimate=0.5):

        self.err_measure = mea_e
        self.err_estimate = est_e
        self.q = q
        self.current_estimate = current_estimate
        self.last_estimate = current_estimate
        self.kalman_gain = 0

    def updateEstimate(self, mea):

        self.kalman_gain = self.err_estimate / (self.err_estimate + self.err_measure)
        self.current_estimate = self.last_estimate + \
            self.kalman_gain * (mea - self.last_estimate)
        self.err_estimate = (1.0 - self.kalman_gain) * self.err_estimate + \
            abs(self.last_estimate-self.current_estimate)*self.q
        self.last_estimate = self.current_estimate

        return self.current_estimate
