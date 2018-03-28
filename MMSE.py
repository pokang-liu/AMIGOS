import itertools
import os
import numpy as np


def MMSE(time_series, M, r, tor):
    m_count = 0
    m_1_count = 0
    channel_num = time_series.shape[0]
    vec_1 = np.zeros(sum(M))
    vec_2 = np.zeros(sum(M))

    len_xm = len(time_series[0]) - max(M) * max(r)
    for i in range(len_xm):
        for j in range(i + 1, len_xm):

            for k in range(channel_num):
                vec_1[:M[i]] = time_series[k][i:i + M[i]]
                vec_2[:M[j]] = time_series[k][j:j + M[j]]
            maxnorm = np.max(np.abs(vec_1 - vec_2))
            if maxnorm < tor:
                m_count += 1
                for c in range(channel_num):
                    diff = abs(time_series[c][i + M[i]] -
                               time_series[c][j + M[j]])
                    if diff < tor:
                        m_1_count += 1
    m_count = m_count / (len_xm * (len_xm - 1))

    m_1_count = m_1_count / (len_xm * (len_xm - 1) * channel_num * channel_num)

    return -log(m_1_count / m_count)
