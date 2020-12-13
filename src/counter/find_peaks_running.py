import numpy as np 
from scipy import signal

class RealtimePeakDetector():
    def __init__(self, lag, filterOrder, threshold):
        self.lag = lag
        self.threshold = threshold
        # initialize array
        self.y = []
        self.filteredY = []
        self.avg = []
        self.std = []
        self.signal = []
        # filter design
        fs = 30; fcut = 0.5; cutoff = 2*fcut/fs
        self.filterParam = signal.firwin(filterOrder, cutoff)

    def thresholding_algo(self, new_value):
        self.y.append(new_value)
        i = len(self.y)-1
        if i==0:
            self.filteredY.append(self.y[i])
            self.avg.append(self.y[i])
            self.std.append(0)
            self.signal.append(0)
            return 0
        if i<self.lag:
            self.filteredY.append(signal.lfilter(self.filterParam, 1.0, self.y)[-1])
            self.avg.append(np.mean(self.filteredY))
            self.std.append(np.std(self.filteredY))
            self.signal.append(0)
            return 0

        self.filteredY.append(signal.lfilter(self.filterParam, 1.0, self.y)[-1])

        if (self.filteredY[i] - self.avg[i-1]) > self.threshold*self.std[i-1]:
            self.signal.append(1)
        else:
            self.signal.append(0)

        # self.filteredY.append(signal.lfilter(self.filterParam, 1.0, self.y)[-1])
        self.avg.append(np.mean(self.filteredY[(i-self.lag):i]))
        self.std.append(np.std(self.filteredY[(i-self.lag):i]))

        return self.signal[i]