#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Utilities
'''
import os
import numpy as np
from PyEMD import EMD
from scipy.signal import butter, lfilter, filtfilt, welch

from config import MISSING_DATA_SUBJECT, SUBJECT_NUM, VIDEO_NUM, FEATURE_NAMES
from sklearn.feature_selection import f_regression

def pvalue(path):
    ''' Calculate P-value '''
    # read extracted features
    amigos_data = np.loadtxt(os.path.join(path, 'features.csv'), delimiter=',')

    # read labels and split to 0 and 1 by
    labels = np.loadtxt(os.path.join(path, 'label.csv'), delimiter=',')
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
    
    _, a_pvalues = f_regression(amigos_data, a_labels)
    _, v_pvalues = f_regression(amigos_data, v_labels)
    
    print('\nUse Arousal Labels')
    print("Number of features (p < 0.05): {}".format(a_pvalues[a_pvalues < 0.05].size))
    for i in range(a_pvalues[a_pvalues < 0.05].size):
        print("Features: {}".format(FEATURE_NAMES[np.where(a_pvalues < 0.05)[0][i]]))
        
    print('\nUse Valence Labels')
    print("Number of features (p < 0.05): {}".format(v_pvalues[v_pvalues < 0.05].size))
    for i in range(a_pvalues[a_pvalues < 0.05].size):
        print("Features: {}".format(FEATURE_NAMES[np.where(v_pvalues < 0.05)[0][i]]))
        
def fisher_idx(features, labels):
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
    Fisher = (abs(mean_features0 - mean_features1)) / std_sum
    Fisher = np.array(Fisher)    
    Fisher_sorted = np.argsort(Fisher)# sort the fisher from small to large
    sorted_feature_idx = Fisher_sorted[::-1]# arrange from large to small        
    return sorted_feature_idx
        
def fisher_selection(h, features, labels):    
    sorted_feature_idx = fisher_idx(features, labels)
    h_features = np.zeros((features.shape[0], h))
    for i in range(features.shape[0]):
        for j in range(h):
            h_features[i][j] = features[i][sorted_feature_idx[j]]
    return h_features

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


def main():
    ''' Main function '''
    # Should write some tests


if __name__ == '__main__':

    main()
