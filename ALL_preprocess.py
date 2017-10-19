#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Functions for Preprocessing
'''

import os
import pickle
import warnings
from biosppy.signals import eeg
from biosppy.signals import ecg
import numpy as np
from scipy.signal import butter, lfilter, filtfilt, detrend
from scipy.stats import skew, kurtosis

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

SUBJECT_NUM = 40
VIDEO_NUM = 16
SAMPLE_RATE = 128
MISSING_DATA = [(9, 1), (9, 2), (9, 3), (9, 6), (9, 7), (9, 9), (9, 11),
                (9, 12), (9, 13), (9, 15), (9, 16), (12, 5), (21, 2), (21, 11),
                (22, 16), (23, 1), (23, 5), (23, 7), (23, 9), (23, 12), (24, 1),
                (24, 8), (24, 12), (24, 13), (24, 14), (24, 15), (24, 16), (33, 1),
                (33, 2), (33, 3), (33, 7), (33, 8), (33, 9), (33, 10), (33, 11),
                (33, 13), (33, 16)]


def filter(spectrum, lower, upper):
    ''' Get lower and upper index '''
    lo_idx = (np.abs(spectrum - lower)).argmin()
    up_idx = (np.abs(spectrum - upper)).argmin()

    return [lo_idx, up_idx]


def spectrum_power(spectrum, idx_pairs):
    ''' Return power of specific range of index '''
    lo_idx = idx_pairs[0]
    up_idx = idx_pairs[1]
    power = 0
    spectrumbuf = spectrum[lo_idx:up_idx + 1]
    for val in np.nditer(spectrumbuf):
        power += abs(val * val)

    return np.array([power])


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


def eeg_preprocessing(signals):
    ''' Preprocessing for EEG signals '''
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

    print("There are {} EEG features".format(features.size))

    features = {
        'theta_power': theta_power,
        'slow_alpha_power': alpha_low_power,
        'alpha_power': alpha_high_power,
        'beta_power': beta_power,
        'gamma_power': gamma_power,
        'theta_spa': theta_spa,
        'slow_alpha_spa': alpha_low_spa,
        'alpha_spa': alpha_high_spa,
        'beta_spa': beta_spa,
        'gamma_spa': gamma_spa
    }

    return features


def ecg_preprocessing(signals):
    ''' Preprocessing for ECG signals '''
    ecg_all = ecg.ecg(signal=signals, sampling_rate=128., show=False)

    rpeaks = ecg_all['rpeaks']

    ecg_fourier = np.fft.fft(signals)
    ecg_freq_idx = np.fft.fftfreq(signals.size, d=(1 / 128))
    positive_ecg_freq_idx = ecg_freq_idx[:(int((ecg_freq_idx.shape[0] + 1) / 2))]

    power_0_6 = np.array([])
    for i in range(60):
        power_0_6 = np.append(power_0_6,
                              spectrum_power(ecg_fourier,
                                             (filter(positive_ecg_freq_idx,
                                                     0 + (i * 0.1),
                                                     0.1 + (i * 0.1)))))

    IBI = np.array([])
    for i in range(len(rpeaks) - 1):
        IBI = np.append(IBI, (rpeaks[i + 1] - rpeaks[i]) / 128)

    heart_rate = 1 / IBI

    mean_IBI = np.mean(IBI)
    rms_IBI = np.sqrt(np.mean(np.square(IBI)))
    std_IBI = np.std(IBI)
    skew_IBI = skew(IBI)
    kurt_IBI = kurtosis(IBI)
    per_above_IBI = IBI[IBI > mean_IBI + std_IBI].size / IBI.size
    per_below_IBI = IBI[IBI < mean_IBI - std_IBI].size / IBI.size

    IBI_fourier = np.fft.fft(IBI)
    IBI_freq_idx = np.fft.fftfreq(IBI.size, d=(mean_IBI / 128))
    positive_IBI_freq_idx = IBI_freq_idx[:(int((IBI_freq_idx.shape[0] + 1) / 2))]

    power_001_008 = spectrum_power(IBI_fourier,
                                   np.array(filter(positive_IBI_freq_idx, 0.01, 0.08)))
    power_008_015 = spectrum_power(IBI_fourier,
                                   np.array(filter(positive_IBI_freq_idx, 0.08, 0.15)))
    power_015_050 = spectrum_power(IBI_fourier,
                                   np.array(filter(positive_IBI_freq_idx, 0.15, 0.5)))

    mean_heart_rate = np.mean(heart_rate)
    std_heart_rate = np.std(heart_rate)
    skew_heart_rate = skew(heart_rate)
    kurt_heart_rate = kurtosis(heart_rate)
    per_above_heart_rate = heart_rate[heart_rate >
                                      mean_heart_rate + std_heart_rate].size / heart_rate.size
    per_below_heart_rate = heart_rate[heart_rate <
                                      mean_heart_rate - std_heart_rate].size / heart_rate.size

    features = np.concatenate(([rms_IBI, mean_IBI], power_0_6,
                               [power_001_008, power_008_015,
                                power_015_050, mean_heart_rate,
                                std_heart_rate, skew_heart_rate,
                                kurt_heart_rate, per_above_heart_rate,
                                per_below_heart_rate, std_IBI, skew_IBI,
                                kurt_IBI, per_above_IBI, per_below_IBI]))

    print("There are {} ECG features".format(features.size))

    features = {
        'rms_IBI': rms_IBI,
        'mean_IBI': mean_IBI,
        'power_0_6': power_0_6,
        'power_001_008': power_001_008,
        'power_008_015': power_008_015,
        'power_015_050': power_015_050,
        'mean_heart_rate': mean_heart_rate,
        'std_heart_rate': std_heart_rate,
        'skew_heart_rate': skew_heart_rate,
        'kurt_heart_rate': kurt_heart_rate,
        'per_above_heart_rate': per_above_heart_rate,
        'per_below_heart_rate': per_below_heart_rate,
        'std_IBI': std_IBI,
        'skew_IBI': skew_IBI,
        'kurt_IBI': kurt_IBI,
        'per_above_IBI': per_above_IBI,
        'per_below_IBI': per_below_IBI
    }

    return features


def gsr_preprocessing(signals):
    ''' Preprocessing for GSR signals '''
    der_signals = np.gradient(signals)
    con_signals = 1 / signals
    nor_con_signals = (con_signals - np.mean(con_signals)) / np.std(con_signals)

    mean = np.mean(signals)
    der_mean = np.mean(der_signals)
    neg_der_mean = np.mean(der_signals[der_signals < 0])
    neg_der_pro = der_signals[der_signals < 0].size / der_signals.size

    local_min = 0
    for i in range(signals.shape[0] - 1):
        if i == 0:
            continue
        if signals[i - 1] > signals[i] and signals[i] < signals[i + 1]:
            local_min += 1

    rising_time = 0
    rising_cnt = 0
    for i in range(der_signals.size - 1):
        if der_signals[i] > 0 and der_signals[i + 1] < 0:
            rising_cnt += 1
        else:
            rising_time += 1

    avg_rising_time = rising_time / (rising_cnt * SAMPLE_RATE)

    gsr_fourier = np.fft.fft(signals)
    gsr_freq_idx = np.fft.fftfreq(signals.size, d=(1 / 128))
    positive_gsr_freq_idx = gsr_freq_idx[:(int((gsr_freq_idx.shape[0] + 1) / 2))]

    power_0_24 = np.array([])
    for i in range(21):
        power_0_24 = np.append(power_0_24,
                               spectrum_power(gsr_fourier,
                                              (filter(positive_gsr_freq_idx,
                                                      0 + (i * 0.8 / 7),
                                                      0.1 + (i * 0.8 / 7)))))

    SCSR = detrend(butter_lowpass_filter(nor_con_signals, 0.2, 128))
    SCVSR = detrend(butter_lowpass_filter(nor_con_signals, 0.08, 128))

    zero_cross_SCSR 	= 0
    zero_cross_SCVSR 	= 0
    peaks_cnt_SCSR 		= 0
    peaks_cnt_SCVSR 	= 0
    peaks_value_SCSR 	= 0.
    peaks_value_SCVSR 	= 0.

    for i in range(nor_con_signals.size - 1):
        if SCSR[i] * SCSR[i + 1] < 0:
            zero_cross_SCSR += 1
        if SCVSR[i] * SCVSR[i + 1] < 0:
            zero_cross_SCVSR += 1
        if i == 0:
            continue
        elif SCSR[i - 1] > SCSR[i] and SCSR[i] < SCSR[i + 1]:
            peaks_value_SCSR += SCSR[i]
            peaks_cnt_SCSR += 1
        elif SCVSR[i - 1] > SCVSR[i] and SCVSR[i] < SCVSR[i + 1]:
            peaks_value_SCVSR += SCSR[i]
            peaks_cnt_SCVSR += 1

    zcr_SCSR = zero_cross_SCSR / (nor_con_signals.size / 128)
    zcr_SCVSR = zero_cross_SCVSR / (nor_con_signals.size / 128)

    mean_peak_SCSR = peaks_value_SCSR / peaks_cnt_SCSR if peaks_cnt_SCSR != 0 else 0
    mean_peak_SCVSR = peaks_value_SCVSR / peaks_cnt_SCVSR if peaks_value_SCVSR != 0 else 0

    features = np.concatenate(([mean, der_mean, neg_der_mean, neg_der_pro,
                                local_min, avg_rising_time], power_0_24,
                               [zcr_SCSR, zcr_SCVSR, mean_peak_SCSR, mean_peak_SCVSR]))

    print("There are {} GSR features".format(features.size))

    features = {
        'mean_sr': mean,
        'mean_sr_der': der_mean,
        'mean_sr_neg_der': neg_der_mean,
        'pro_neg_der': neg_der_pro,
        'local_min_gsr': local_min,
        'avg_rising_time': avg_rising_time,
        'power_0_24': power_0_24,
        'zcr_SCSR': zcr_SCSR,
        'zcr_SCVSR': zcr_SCVSR,
        'mean_peak_SCSR': mean_peak_SCSR,
        'mean_peak_SCVSR': mean_peak_SCVSR
    }

    return features


def read_dataset(path):
    ''' Read AMIGOS dataset '''
    amigos_data = dict()

    for sid in range(SUBJECT_NUM):
        for vid in range(VIDEO_NUM):
            if (sid + 1, vid + 1) in MISSING_DATA:
                continue
            print('Reading {}_{}.csv'.format(sid + 1, vid + 1))
            signals = np.genfromtxt(os.path.join(path, "{}_{}.csv".format(sid + 1, vid + 1)),
                                    delimiter=',')
            eeg_signals = signals[:, :14]
            ecg_signals = signals[:, 14]  # Column 14 or 15
            gsr_signals = signals[:, -1]

            eeg_features = eeg_preprocessing(eeg_signals)
            ecg_features = ecg_preprocessing(ecg_signals)
            gsr_features = gsr_preprocessing(gsr_signals)

            features = {**eeg_features, **ecg_features, **gsr_features}
            amigos_data["{}_{}".format(sid + 1, vid + 1)] = features

    return amigos_data


def main():
    ''' Main function '''
    amigos_data = read_dataset('data')

    with open(os.path.join('data', 'features.p'), 'wb') as pickle_file:
        pickle.dump(amigos_data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    main()
