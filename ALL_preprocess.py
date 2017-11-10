#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Functions for Preprocessing
'''

from math import log
import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals import ecg
from scipy.stats import skew, kurtosis

from utils import butter_highpass_filter, butter_lowpass_filter, getfreqs_power, getBand_Power, getFiveBands_Power, detrend

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

SUBJECT_NUM = 40
VIDEO_NUM = 16
SAMPLE_RATE = 128.
MISSING_DATA = [(9, 1), (9, 2), (9, 3), (9, 6), (9, 7), (9, 9), (9, 11),
                (9, 12), (9, 13), (9, 15), (9, 16), (12, 5), (21, 2), (21, 11),
                (22, 16), (23, 1), (23, 5), (23, 7), (23, 9), (23, 12), (24, 1),
                (24, 8), (24, 12), (24, 13), (24, 14), (24, 15), (24, 16), (33, 1),
                (33, 2), (33, 3), (33, 7), (33, 8), (33, 9), (33, 10), (33, 11),
                (33, 13), (33, 16)]


def eeg_preprocessing(signals):
    ''' Preprocessing for EEG signals '''
    trans_signals = np.transpose(signals)

    theta_power = []
    slow_alpha_power = []
    alpha_power = []
    beta_power = []
    gamma_power = []
    psd_list = [theta_power, slow_alpha_power, alpha_power, beta_power, gamma_power]

    theta_spec_power = []
    slow_alpha_spec_power = []
    alpha_spec_power = []
    beta_spec_power = []
    gamma_spec_power = []
    spec_power_list = [theta_spec_power, slow_alpha_spec_power,
                       alpha_spec_power, beta_spec_power, gamma_spec_power]

    theta_spa = []
    slow_alpha_spa = []
    alpha_spa = []
    beta_spa = []
    gamma_spa = []

    for channel_signals in trans_signals:
        freqs, power = getfreqs_power(channel_signals, fs=128., nperseg=128, scaling='density')
        psd = getFiveBands_Power(freqs, power)
        for band, band_list in zip(psd, psd_list):
            band_list.append(log(band))

        freqs_, power_ = getfreqs_power(channel_signals, fs=128., nperseg=128, scaling='spectrum')
        spec_power = getFiveBands_Power(freqs_, power_)
        for band, band_list in zip(spec_power, spec_power_list):
            band_list.append(band)

    for i in range(7):
        theta_spa.append((theta_spec_power[i] - theta_spec_power[13 - i]) /
                         (theta_spec_power[i] + theta_spec_power[13 - i]))
        slow_alpha_spa.append((slow_alpha_spec_power[i] - slow_alpha_spec_power[13 - i]) /
                              (slow_alpha_spec_power[i] + slow_alpha_spec_power[13 - i]))
        alpha_spa.append((alpha_spec_power[i] - alpha_spec_power[13 - i]) /
                         (alpha_spec_power[i] + alpha_spec_power[13 - i]))
        beta_spa.append((beta_spec_power[i] - beta_spec_power[13 - i]) /
                        (beta_spec_power[i] + beta_spec_power[13 - i]))
        gamma_spa.append((gamma_spec_power[i] - gamma_spec_power[13 - i]) /
                         (gamma_spec_power[i] + gamma_spec_power[13 - i]))

    features = {
        'theta_power': theta_power,
        'slow_alpha_power': slow_alpha_power,
        'alpha_power': alpha_power,
        'beta_power': beta_power,
        'gamma_power': gamma_power,
        'theta_spa': theta_spa,
        'slow_alpha_spa': slow_alpha_spa,
        'alpha_spa': alpha_spa,
        'beta_spa': beta_spa,
        'gamma_spa': gamma_spa
    }

    return features


def ecg_preprocessing(signals):
    ''' Preprocessing for ECG signals '''
    # some data have high peak value due to noise
    # signals , _ = detrend(signals)
    signals= butter_highpass_filter(signals, 1.0, 256.0)
    ecg_all = ecg.ecg(signal=signals, sampling_rate=256., show=False)
    rpeaks = ecg_all['rpeaks']  # R-peak location indices.

    # ECG
    freqs, power = getfreqs_power(signals, fs=256., nperseg=signals.size, scaling='spectrum')
    power_0_6 = []
    for i in range(60):
        power_0_6.append(getBand_Power(freqs, power, lower=0 + (i * 0.1), upper=0.1 + (i * 0.1)))

    IBI = np.array([])
    for i in range(len(rpeaks) - 1):
        IBI = np.append(IBI, (rpeaks[i + 1] - rpeaks[i]) / 128.0)

    heart_rate = np.array([])
    for i in range(len(IBI)):
        append_value = 60.0 / IBI[i] if IBI[i] != 0 else 0
        heart_rate = np.append(heart_rate, append_value)

    mean_IBI = np.mean(IBI)
    rms_IBI = np.sqrt(np.mean(np.square(IBI)))
    std_IBI = np.std(IBI)
    skew_IBI = skew(IBI)
    kurt_IBI = kurtosis(IBI)
    per_above_IBI = float(IBI[IBI > mean_IBI + std_IBI].size) / float(IBI.size)
    per_below_IBI = float(IBI[IBI < mean_IBI - std_IBI].size) / float(IBI.size)

    # IBI
    freqs_, power_ = getfreqs_power(IBI, fs=1.0 / mean_IBI, nperseg=IBI.size, scaling='spectrum')
    power_000_004 = getBand_Power(freqs_, power_, lower=0., upper=0.04) #VLF
    power_004_015 = getBand_Power(freqs_, power_, lower=0.04, upper=0.15) #LF
    power_015_040 = getBand_Power(freqs_, power_, lower=0.15, upper=0.40) #HF
    power_000_040 = getBand_Power(freqs_, power_, lower=0., upper=0.40) #TF
    LF_HF = power_004_015/power_015_040
    LF_TF = power_004_015/power_000_040
    HF_TF = power_015_040/power_000_040    

    mean_heart_rate = np.mean(heart_rate)
    std_heart_rate = np.std(heart_rate)
    skew_heart_rate = skew(heart_rate)
    kurt_heart_rate = kurtosis(heart_rate)
    per_above_heart_rate = float(heart_rate[heart_rate >
                                            mean_heart_rate + std_heart_rate].size) / float(heart_rate.size)
    per_below_heart_rate = float(heart_rate[heart_rate <
                                            mean_heart_rate - std_heart_rate].size) / float(heart_rate.size)

    features = {
        'rms_IBI': rms_IBI,
        'mean_IBI': mean_IBI,
        'power_0_6': power_0_6,
        'power_000_004': power_000_004,
        'power_004_015': power_004_015,
        'power_015_040': power_015_040,
        'LF_HF': LF_HF,
        'LF_TF': LF_TF,
        'HF_TF': HF_TF,
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
    con_signals = 1.0 / signals
    nor_con_signals = (con_signals - np.mean(con_signals)) / np.std(con_signals)

    mean = np.mean(signals)
    der_mean = np.mean(der_signals)
    neg_der_mean = np.mean(der_signals[der_signals < 0])
    neg_der_pro = float(der_signals[der_signals < 0].size) / float(der_signals.size)

    local_min = 0
    for i in range(signals.shape[0] - 1):
        if i == 0:
            continue
        if signals[i - 1] > signals[i] and signals[i] < signals[i + 1]:
            local_min += 1

    # Using SC calculates rising time
    det_nor_signals, trend = detrend(nor_con_signals)
    lp_det_nor_signals = butter_lowpass_filter(det_nor_signals, 0.5, 128)
    der_lp_det_nor_signals = np.gradient(lp_det_nor_signals)

    rising_time = 0
    rising_cnt = 0
    for i in range(der_lp_det_nor_signals.size - 1):
        if der_lp_det_nor_signals[i] > 0:
            rising_time += 1
            if der_lp_det_nor_signals[i + 1] < 0:
                rising_cnt += 1

    avg_rising_time = rising_time*(1./128.) / rising_cnt

    freqs, power = getfreqs_power(signals, fs=128., nperseg=signals.size, scaling='spectrum')
    power_0_24 = []
    for i in range(21):
        power_0_24.append(getBand_Power(freqs, power, lower=0 +
                                        (i * 0.8 / 7), upper=0.1 + (i * 0.8 / 7)))

    SCSR, _ = detrend(butter_lowpass_filter(nor_con_signals, 0.2, 128))
    SCVSR, _ = detrend(butter_lowpass_filter(nor_con_signals, 0.08, 128))

    zero_cross_SCSR = 0
    zero_cross_SCVSR = 0
    peaks_cnt_SCSR = 0
    peaks_cnt_SCVSR = 0
    peaks_value_SCSR = 0.
    peaks_value_SCVSR = 0.

    zc_idx_SCSR = np.array([], int)  # must be int, otherwise it will be float
    zc_idx_SCVSR = np.array([], int)
    for i in range(nor_con_signals.size - 1):
        if SCSR[i] * next((j for j in SCSR[i + 1:] if j != 0), 0) < 0:
            zero_cross_SCSR += 1
            zc_idx_SCSR = np.append(zc_idx_SCSR, i + 1)
        if SCVSR[i] * next((j for j in SCVSR[i + 1:] if j != 0), 0) < 0:
            zero_cross_SCVSR += 1
            zc_idx_SCVSR = np.append(zc_idx_SCVSR, i)

    for i in range(zc_idx_SCSR.size - 1):
        peaks_value_SCSR += np.absolute(SCSR[zc_idx_SCSR[i]:zc_idx_SCSR[i + 1]]).max()
        peaks_cnt_SCSR += 1
    for i in range(zc_idx_SCVSR.size - 1):
        peaks_value_SCVSR += np.absolute(SCVSR[zc_idx_SCVSR[i]:zc_idx_SCVSR[i + 1]]).max()
        peaks_cnt_SCVSR += 1

    zcr_SCSR = zero_cross_SCSR / (nor_con_signals.size / 128.)
    zcr_SCVSR = zero_cross_SCVSR / (nor_con_signals.size / 128.)

    mean_peak_SCSR = peaks_value_SCSR / peaks_cnt_SCSR if peaks_cnt_SCSR != 0 else 0
    mean_peak_SCVSR = peaks_value_SCVSR / peaks_cnt_SCVSR if peaks_value_SCVSR != 0 else 0

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
                print("Skipping {}_{}.csv".format(sid + 1, vid + 1))
                continue
            print('Reading {}_{}.csv'.format(sid + 1, vid + 1))
            signals = np.genfromtxt(os.path.join(path, "{}_{}.csv".format(sid + 1, vid + 1)),
                                    delimiter=',')
            eeg_signals = signals[:, :14]
            ecg_signals = signals[:, 14]  # Column 14 or 15
            gsr_signals = signals[20:, -1]  # ignore the first 20 data, since there is noise in it

            eeg_features = eeg_preprocessing(eeg_signals)
            ecg_features = ecg_preprocessing(ecg_signals)
            gsr_features = gsr_preprocessing(gsr_signals)

            features = {
                'eeg': eeg_features,
                'ecg': ecg_features,
                'gsr': gsr_features
            }
            amigos_data["{}_{}".format(sid + 1, vid + 1)] = features

    return amigos_data


def main():
    ''' Main function '''
    amigos_data = read_dataset('../AMIGOS_data')
    with open(os.path.join('data', 'features.p'), 'wb') as pickle_file:
        pickle.dump(amigos_data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    main()
