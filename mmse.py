import itertools
import os
import numpy as np
from math import log
'''
Multivariate Multi-Scale Entropy implementation
'''
def MMSE(time_series, M, r, tor):
    m_count = 0.0
    m_1_count = 0.0
    channel_num = time_series.shape[0]
    vec_1 = np.zeros(sum(M))
    vec_2 = np.zeros(sum(M))

    len_xm = len(time_series[0]) - max(M) * max(r)
    for i in range(len_xm):
        for j in range(i + 1, len_xm):

            for k in range(channel_num - 1):
                vec_1[sum(M[:k]):sum(M[:k + 1])] = time_series[k][i:i + M[k]]
                vec_2[sum(M[:k]):sum(M[:k + 1])] = time_series[k][j:j + M[k]]
            maxnorm = np.max(np.abs(vec_1 - vec_2))
            if maxnorm < tor:
                m_count += 1
                for c in range(channel_num):
                    diff = abs(time_series[c][i + M[c]] - time_series[c][j + M[c]])
                    if diff < tor:
                        m_1_count += 1
    m_count = m_count / (len_xm * (len_xm - 1))

    m_1_count = m_1_count / (len_xm * (len_xm - 1) * channel_num * channel_num)

    return -log((float(m_1_count + 0.00001)) / float(m_count + 0.00001))


signals = np.genfromtxt(os.path.join('data', "1_1.csv"), delimiter=',')
eeg_signals = np.transpose(signals[:, :3])
grained_signals = []

from utils import util_granulate_time_series

for c in range(eeg_signals.shape[0]):
    grained_signals.append(util_granulate_time_series(eeg_signals[0], 10))

grained_signals = np.array(grained_signals)
print(grained_signals.shape)

print(MMSE(grained_signals, [2, 2, 2], [1, 1, 1], 0.15))
