#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Functions for Preprocessing
'''

import os
from biosppy.signals import eeg
from biosppy.signals import ecg
import numpy as np
from scipy.signal import butter, lfilter, freqz, filtfilt, detrend

PARTICIPANT_NUM = 1
VIDEO_NUM = 1
SAMPLE_RATE = 128


def filter(spectrum, lower, upper):
    ''' Filter '''
    lo_idx = (np.abs(spectrum - lower)).argmin()
    up_idx = (np.abs(spectrum - upper)).argmin()

    return [lo_idx, up_idx]


def power(spectrum, idx_pairs):
    ''' Power '''
    lo_idx = idx_pairs[0]
    up_idx = idx_pairs[1]
    power = 0
    spectrumbuf = spectrum[lo_idx:up_idx + 1]
    for x in np.nditer(spectrumbuf):
        power += abs(x * x)

    return np.array([power])


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    for i in range(data.shape[1]):
        data[:, i] = filtfilt(b, a, data[:, i])

    return data


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def eeg_preprocessing(signals):
    ''' Preprocessing for EEG signals '''
    signals = signals - np.mean(signals[:, :14], axis=0)
    # signals = butter_highpass_filter(signals, 2, 128)

    eeg_all = eeg.eeg(signal=signals, sampling_rate=128., show=False)

    theta = eeg_all['theta']
    alpha_low = eeg_all['alpha_low']
    alpha_high = eeg_all['alpha_high']
    beta = eeg_all['beta']
    gamma = eeg_all['gamma']

    theta_power = np.sum(theta, axis=0)
    alpha_low_power = np.sum(alpha_low, axis=0)
    alpha_high_power = np.sum(alpha_high, axis=0)
    beta_power = np.sum(beta, axis=0)
    gamma_power = np.sum(gamma, axis=0)

    theta_spa = []
    alpha_low_spa = []
    alpha_high_spa = []
    beta_spa = []
    gamma_spa = []

    for i in range(7):
        theta_spa.append((theta_power[i] - theta_power[13 - i]) /
                         (theta_power[i] + theta_power[13 - i]))
        alpha_low_spa.append((alpha_low_power[i] - alpha_low_power[13 - i]) /
                             (alpha_low_power[i] + alpha_low_power[13 - i]))
        alpha_high_spa.append((alpha_high_power[i] - alpha_high_power[13 - i]) /
                              (alpha_high_power[i] + alpha_high_power[13 - i]))
        beta_spa.append((beta_power[i] - beta_power[13 - i]) /
                        (beta_power[i] + beta_power[13 - i]))
        gamma_spa.append((gamma_power[i] - gamma_power[13 - i]) /
                         (gamma_power[i] + gamma_power[13 - i]))

    features = np.concatenate((theta_power, alpha_low_power,
                               alpha_high_power, beta_power,
                               gamma_power, theta_spa, alpha_low_spa,
                               alpha_high_spa, beta_spa, gamma_spa))

    return features


def ecg_preprocessing(signals):
    ''' Preprocessing for ECG signals '''
    # ecg_all = ecg.ecg(signal=signals, sampling_rate=128., show=False)

    # rpeaks = ecg_all['rpeaks']

    # ecg_fourier = np.fft.fft(signals)

    # ecg_freq_idx = np.fft.fftfreq(signals.size, d=1 / 128)
    # positive_ecg_freq_idx = ecg_freq_idx[:(
    #     int((ecg_freq_idx.shape[0] + 1) / 2))]

    # power_0_6 = np.array([])
    # for i in range(60):
    #     power_0_6 = np.append(power_0_6,
    #                           power(ecg_fourier,
    #                                 (filter(positive_ecg_freq_idx,
    #                                         0 + (i * 0.1),
    #                                         0.1 + (i * 0.1)))))

    return []


def gsr_preprocessing(signals):
    ''' Preprocessing for GSR signals '''
    der_signals = np.gradient(signals)
    nor_signals = (signals - np.mean(signals)) / np.std(signals)
    detrend_signals = detrend(signals)

    mean = np.mean(signals)
    der_mean = np.mean(der_signals)
    neg_der_mean = np.mean(der_signals[der_signals < 0])
    neg_der_por = der_signals[der_signals < 0].size / der_signals.size

    local_min = 0
    for idx, signal in enumerate(signals):
        if idx == 0:
            continue
        if signals[idx - 1] > signal and signal < signals[idx + 1]:
            local_min += 1

    rising_time = 0
    rising_ctn = 0
    for _, signal in enumerate(der_signals):
        if signal < 0:
            rising_ctn += 1
        else:
            rising_time += 1

    avg_rising_time = rising_time / (rising_ctn * SAMPLE_RATE)

    return []


def read_dataset(path):
    ''' Read AMIGOS dataset '''
    amigos_data = None

    for pid in range(PARTICIPANT_NUM):
        for vid in range(VIDEO_NUM):
            signals = np.genfromtxt(os.path.join(path, "{}_{}.csv".format(pid + 1, vid + 1)),
                                    delimiter=',')
            eeg_signals = signals[:, :14]
            ecg_signals = signals[:, 14:16]
            gsr_signals = signals[:, -1]

            eeg_features = eeg_preprocessing(eeg_signals)
            ecg_features = ecg_preprocessing(ecg_signals)
            gsr_features = gsr_preprocessing(gsr_signals)

            features = np.concatenate(
                (eeg_features, ecg_features, gsr_features))
            amigos_data = features if amigos_data is None else np.vstack(
                (amigos_data, features))

    return amigos_data


def main():
    ''' Main function '''
    amigos_data = read_dataset('data')

    print(amigos_data.shape)
    print(amigos_data[0])

    # np.save('amigos_data', amigos_data)


if __name__ == '__main__':

    main()
