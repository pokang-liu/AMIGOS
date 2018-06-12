#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Utilities
'''

import os

import numpy as np
from PyEMD import EMD
from scipy.signal import butter, lfilter, filtfilt, welch
from sklearn.feature_selection import f_classif

from config import MISSING_DATA_SUBJECT, SUBJECT_NUM

FEATURE_NAMES = []

# for s in range(1, 21):
#     for d in range(2, 7):
#         for r in range(1, 6):
#             FEATURE_NAMES.append("eeg_mmpe_s{}_d{}_r{}".format(s, d, r))

# for s in range(1, 4):
#     for d in range(2, 7):
#         FEATURE_NAMES.append("ecg_rcmpe_s{}_d{}".format(s, d))

for s in range(1, 21):
    for d in range(2, 7):
        FEATURE_NAMES.append("gsr_rcmpe_s{}_d{}".format(s, d))


def pvalue(path):
    ''' Calculate P-value '''
    # read extracted features
    amigos_data = np.loadtxt(os.path.join(path, 'mpe', 'mpe_features.csv'), delimiter=',')
    # amigos_data = amigos_data[:, :500] # EEG
    # amigos_data = amigos_data[:, 500:515] # ECG
    amigos_data = amigos_data[:, 515:] # GSR

    # read labels and split to 0 and 1 by
    a_labels, v_labels = read_labels(os.path.join(path, 'label.csv'))

    # calculate p-value
    _, a_pvalues = f_classif(amigos_data, a_labels)
    _, v_pvalues = f_classif(amigos_data, v_labels)

    # arousal
    sel_idx = np.argsort(a_pvalues)[:20]
    a_saved_name = []
    for idx in sel_idx:
        a_saved_name.append(FEATURE_NAMES[idx])

    # valence
    sel_idx = np.argsort(v_pvalues)[:0]
    v_saved_name = []
    for idx in sel_idx:
        v_saved_name.append(FEATURE_NAMES[idx])
    with open('data/s_gsr_rcmpe_name', 'w') as f:
        for name in a_saved_name:
            f.write("{}\n".format(name))
        for name in v_saved_name:
            f.write("{}\n".format(name))

    print('Arousal')
    for idx in np.argsort(a_pvalues)[:3]:
        print(FEATURE_NAMES[idx], a_pvalues[idx])

    print('Valence')
    for idx in np.argsort(v_pvalues)[:3]:
        print(FEATURE_NAMES[idx], v_pvalues[idx])

    print('\nUse Arousal Labels')
    print("Number of features (p < 0.05): {}".format(a_pvalues[a_pvalues < 0.05].size))
    for i in range(a_pvalues[a_pvalues < 0.05].size):
        print("Features: {}, Value: {:.4f}".format(FEATURE_NAMES[np.where(
            a_pvalues < 0.05)[0][i]], a_pvalues[np.where(a_pvalues < 0.05)[0][i]]))

    print('\nUse Valence Labels')
    print("Number of features (p < 0.05): {}".format(v_pvalues[v_pvalues < 0.05].size))
    for i in range(v_pvalues[v_pvalues < 0.05].size):
        print("Features: {}, Value: {:.4f}".format(FEATURE_NAMES[np.where(
            v_pvalues < 0.05)[0][i]], v_pvalues[np.where(v_pvalues < 0.05)[0][i]]))


def fisher_idx(num, features, labels):
    ''' Get idx sorted by fisher linear discriminant '''
    labels = np.array(labels)
    labels0 = np.where(labels < 1)
    labels1 = np.where(labels > 0)
    labels0 = np.array(labels0).flatten()
    labels1 = np.array(labels1).flatten()
    features0 = np.delete(features, labels1, axis=0)
    features1 = np.delete(features, labels0, axis=0)
    mean_features0 = np.mean(features0, axis=0)
    mean_features1 = np.mean(features1, axis=0)
    std_features0 = np.std(features0, axis=0)
    std_features1 = np.std(features1, axis=0)
    std_sum = std_features1**2 + std_features0**2
    fisher = (abs(mean_features0 - mean_features1)) / std_sum
    fisher_sorted = np.argsort(np.array(fisher))  # sort the fisher from small to large
    sorted_feature_idx = fisher_sorted[::-1]  # arrange from large to small
    return sorted_feature_idx[:num]


def butter_highpass_filter(data, cutoff, fs, order=5):
    ''' Highpass filter '''
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff, fs, order=5):
    ''' Lowpass filter '''
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def getfreqs_power(signals, fs, nperseg, scaling):
    ''' Calculate power density or power spectrum density '''
    if scaling == "density":
        freqs, power = welch(signals, fs=fs, nperseg=nperseg, scaling='density')
        return freqs, power
    elif scaling == "spectrum":
        freqs, power = welch(signals, fs=fs, nperseg=nperseg, scaling='spectrum')
        return freqs, power
    else:
        return 0, 0


def getBand_Power(freqs, power, lower, upper):
    ''' Sum band power within desired frequency range '''
    low_idx = np.array(np.where(freqs <= lower)).flatten()
    up_idx = np.array(np.where(freqs > upper)).flatten()
    band_power = np.sum(power[low_idx[-1]:up_idx[0]])

    return band_power


def getFiveBands_Power(freqs, power):
    ''' Calculate 5 bands power '''
    theta_power = getBand_Power(freqs, power, 3, 7)
    slow_alpha_power = getBand_Power(freqs, power, 8, 10)
    alpha_power = getBand_Power(freqs, power, 8, 13)
    beta_power = getBand_Power(freqs, power, 14, 29)
    gamma_power = getBand_Power(freqs, power, 30, 47)

    return theta_power, slow_alpha_power, alpha_power, beta_power, gamma_power


def detrend(data):
    ''' Detrend data with EMD '''
    emd = EMD()
    imfs = emd(data)
    detrended = np.sum(imfs[:int(imfs.shape[0] / 2)], axis=0)
    trend = np.sum(imfs[int(imfs.shape[0] / 2):], axis=0)

    return detrended, trend


def read_labels(path):
    """ Read labels of arousal and valance

    Arguments:
        path: path of the label file
    Return:
        a_labels: arousal labels
        v_labels: valance labels
    """
    labels = np.loadtxt(path, delimiter=',')
    labels = labels[:, :2]
    a_labels, v_labels = [], []
    for i in range(SUBJECT_NUM):
        if i + 1 in MISSING_DATA_SUBJECT:
            continue
        a_labels_mean = np.mean(labels[i * 16:i * 16 + 16, 0])
        v_labels_mean = np.mean(labels[i * 16:i * 16 + 16, 1])
        for idx, label in enumerate(labels[i * 16:i * 16 + 16, :]):
            a_tmp = 1 if label[0] > a_labels_mean else 0
            v_tmp = 1 if label[1] > v_labels_mean else 0
            a_labels.append(a_tmp)
            v_labels.append(v_tmp)
    a_labels, v_labels = np.array(a_labels), np.array(v_labels)

    return a_labels, v_labels


def sample_entropy(time_series, sample_length, tolerance=None):
    epsilon = 0.000001
    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)

    n = len(time_series)
    prev = np.zeros(n)
    curr = np.zeros(n)
    A = np.zeros((sample_length, 1))  # number of matches for m = [1,...,template_length - 1]
    B = np.zeros((sample_length, 1))  # number of matches for m = [1,...,template_length]

    for i in range(n - 1):
        nj = n - i - 1
        ts1 = time_series[i]
        for jj in range(nj):
            j = jj + i + 1
            if abs(time_series[j] - ts1) < tolerance:  # distance between two vectors
                curr[jj] = prev[jj] + 1
                temp_ts_length = min(sample_length, curr[jj])
                for m in range(int(temp_ts_length)):
                    A[m] += 1
                    if j < n - 1:
                        B[m] += 1
            else:
                curr[jj] = 0
        for j in range(nj):
            prev[j] = curr[j]

    N = n * (n - 1) / 2
    B = np.vstack(([N], B[:sample_length - 1]))
    similarity_ratio = (A) / B
    se = -np.log(similarity_ratio)
    se = np.reshape(se, -1)
    return se


def multiscale_entropy(time_series, scaling_factor, m, tolerance=None):

    n = len(time_series)
    mse = np.zeros((1, scaling_factor))

    for i in range(scaling_factor):
        b = int(np.fix(n / (i + 1)))
        temp_ts = [0] * int(b)
        for j in range(b):
            num = sum(time_series[j * (i + 1): (j + 1) * (i + 1)])
            den = i + 1
            temp_ts[j] = float(num) / float(den)

        se = sample_entropy(temp_ts, m, tolerance)
        mse[0, i] = se[-1]

    return mse[0]


def util_granulate_time_series(time_series, scale):
    """Extract coarse-grained time series
    Args:
        time_series: Time series
        scale: Scale factor
    Returns:
        Vector of coarse-grained time series with given scale factor
    """
    n = len(time_series)
    b = int(np.fix(n / scale))
    cts = [0] * b
    for i in range(b):
        cts[i] = np.mean(time_series[i * scale: (i + 1) * scale])
    return cts


def composite_multiscale_entropy(time_series, m, scale, tolerance=None):
    """Calculate the Composite Multiscale Entropy of the given time series.
    Args:
        time_series: Time series for analysis
        sample_length: Number of sequential points of the time series
        scale: Scale factor
        tolerance: Tolerance (default = 0.1...0.2 * std(time_series))
    Returns:
        Vector containing Composite Multiscale Entropy
    Reference:
        [1] Wu, Shuen-De, et al. "Time series analysis using
            composite multiscale entropy." Entropy 15.3 (2013): 1069-1084.
    """
    cmse = np.zeros((1, scale))

    for i in range(scale):
        for j in range(i):
            tmp = util_granulate_time_series(time_series[j:], j + 1)
            tmpse = sample_entropy(tmp, m, tolerance) / (i + 1)
            cmse[i] += tmpse[-1]

    return cmse


def RC_composite_multiscale_entropy(time_series, sample_length, scale, m, tolerance=None):
    """Calculate the Composite Multiscale Entropy of the given time series.
    Args:
        time_series: Time series for analysis
        sample_length: Number of sequential points of the time series
        scale: Scale factor
        m: equal to sample length
        tolerance: Tolerance (default = 0.1...0.2 * std(time_series))
    Returns:
        Vector containing RC Composite Multiscale Entropy
    Reference:
        [1] Wu, Shuen-De, et al. "Time series analysis using
            composite multiscale entropy." Entropy 15.3 (2013): 1069-1084.
    """
    A_sum = 0
    B_sum = 0
    epsilon = 0.0000001
    for i in range(scale):
        tmp = util_granulate_time_series(time_series[i:], scale)
        A_B = RC_sample_entropy(tmp, sample_length, tolerance)
        # print(A_B)
        B_sum += A_B[m + sample_length - 1][0]
        A_sum += A_B[m - 1][0]
    rcmse = - np.log((A_sum) / B_sum)
    return rcmse


def RC_sample_entropy(time_series, sample_length, tolerance=None):
    """Calculate and return Sample Entropy of the given time series.
    Distance between two vectors defined as Euclidean distance and can
    be changed in future releases
    Args:
        time_series: Vector or string of the sample data
        sample_length (m): Number of sequential points of the time series
        tolerance: Tolerance (default = 0.1...0.2 * std(time_series))
    Returns:
        Vector containing RC Sample Entropy (float)
    References:
        [1] http://en.wikipedia.org/wiki/Sample_Entropy
        [2] http://physionet.incor.usp.br/physiotools/sampen/
        [3] Madalena Costa, Ary Goldberger, CK Peng. Multiscale entropy analysis
            of biological signals
    """
    if tolerance is None:
        tolerance = 0.1 * np.std(time_series)

    n = len(time_series)
    prev = np.zeros(n)
    curr = np.zeros(n)
    A = np.zeros((sample_length, 1))  # number of matches for m = [1,...,template_length - 1]
    B = np.zeros((sample_length, 1))  # number of matches for m = [1,...,template_length]

    for i in range(n - 1):
        nj = n - i - 1
        ts1 = time_series[i]
        for jj in range(nj):
            j = jj + i + 1
            if abs(time_series[j] - ts1) < tolerance:  # distance between two vectors
                curr[jj] = prev[jj] + 1
                temp_ts_length = min(sample_length, curr[jj])
                for m in range(int(temp_ts_length)):
                    A[m] += 1
                    if j < n - 1:
                        B[m] += 1
            else:
                curr[jj] = 0
        for j in range(nj):
            prev[j] = curr[j]

    N = n * (n - 1) / 2
    B = np.vstack(([N], B[:sample_length - 1]))
    A_B = np.vstack((A, B))

    return A_B


def main():
    ''' Main function '''
    # Should write some tests
    pvalue('data')


if __name__ == '__main__':

    main()
