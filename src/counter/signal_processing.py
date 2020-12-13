import scipy.signal
import numpy as np

fs = 10; fcut = 0.5; cutoff = 2*fcut/fs
filter_order = 40
filterParam = scipy.signal.firwin(filter_order, cutoff)

def lfilter_smooth(data):
    data = np.array(data)
    data = scipy.signal.lfilter(filterParam, 1.0, data)
    return data

def avg_smooth(x, window_len=11):
    if window_len < 3:
        return x
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    w = np.ones(window_len, 'd')
    y = np.convolve(w/w.sum(), s, mode='valid')
    y = y[window_len-1:]
    return y

def find_peaks(data):
    peaks = scipy.signal.find_peaks(data, distance=5, prominence=0.1)
    # distance: min distance between 2 samples by x-axis
    # prominence: min difference between peaks and neighbors by y-axis
    return peaks[0]

    