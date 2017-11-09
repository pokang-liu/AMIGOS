#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Utilities
'''

from biosppy.signals import tools
import numpy as np
from PyEMD import EMD
from scipy.signal import butter, lfilter, filtfilt, welch


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def getfreqs_power(signals, fs, nperseg, scaling):
    if scaling == "density":
        freqs, power = welch(signals, fs=fs, nperseg=nperseg, scaling='density')
        return freqs, power
    elif scaling == "spectrum":
        freqs, power = welch(signals, fs=fs, nperseg=nperseg, scaling='spectrum')
        return freqs, power
    else:
        return 0, 0


def getBand_Power(freqs, power, lower, upper):
    Band_power = float(np.array
                       ((tools.band_power(freqs=freqs, power=power,
                                          frequency=[lower, upper], decibel=False)))
                       .flatten())
    return Band_power


def getFiveBands_Power(freqs, power):
    theta_power = getBand_Power(freqs, power, 3, 7)
    slow_alpha_power = getBand_Power(freqs, power, 8, 10)
    alpha_power = getBand_Power(freqs, power, 8, 13)
    beta_power = getBand_Power(freqs, power, 14, 29)
    gamma_power = getBand_Power(freqs, power, 30, 47)

    return theta_power, slow_alpha_power, alpha_power, beta_power, gamma_power


def detrend(data):
    ''' Detrend data with EMD '''
    emd = EMD()
    IMFs = emd(data)
    detrended = np.sum(IMFs[:int(IMFs.shape[0] / 2)], axis=0)
    trend = np.sum(IMFs[int(IMFs.shape[0] / 2):], axis=0)

    return detrended, trend

def main():
    ''' Main function '''
    # Should write some tests


if __name__ == '__main__':

    main()
