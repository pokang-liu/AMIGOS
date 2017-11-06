#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Functions for Preprocessing
'''

import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from math import log
from biosppy.signals import ecg, tools
from scipy.signal import butter, lfilter, filtfilt, detrend, welch
from scipy.stats import skew, kurtosis

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
    
def getfreqs_power(signals,fs):
    spectrum     = tools.power_spectrum(signal=signals, sampling_rate=fs, decibel=False)
    freqs        = spectrum['freqs']
    power        = spectrum['power']    
    return freqs, power

def getBand_Power(freqs,power,lower,upper):
    Band_power = float(np.array
                        ((tools.band_power(freqs=freqs,power=power,frequency=[lower, upper],decibel=False)))
                            .flatten())
    return Band_power
    
def getFiveBands_Power(freqs,power):
    theta_power = getBand_Power(freqs,power,3,7)
    slow_alpha_power = getBand_Power(freqs,power,8,10)
    alpha_power = getBand_Power(freqs,power,8,13)
    beta_power = getBand_Power(freqs,power,14,29)
    gamma_power = getBand_Power(freqs,power,30,47)

    return theta_power, slow_alpha_power, alpha_power, beta_power, gamma_power 

    
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
    spec_power_list = [theta_spec_power, slow_alpha_spec_power, alpha_spec_power, beta_spec_power, gamma_spec_power]

    theta_spa = []
    slow_alpha_spa = []
    alpha_spa = []
    beta_spa = []
    gamma_spa = []

    for channel_signals in trans_signals:        
        freqs, power = welch(channel_signals, fs=128., nperseg=128, scaling='density')
        psd = getFiveBands_Power(freqs,power)
        for band,band_list in zip(psd,psd_list):
            band_list.append(log(band))

        freqs_, power_ = welch(channel_signals, fs=128., nperseg=128, scaling='spectrum')
        spec_power = getFiveBands_Power(freqs_,power_)
        for band,band_list in zip(spec_power,spec_power_list):
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

    # features = np.concatenate((theta_power, slow_alpha_power,
    #                            alpha_power, beta_power,
    #                            gamma_power, theta_spa, slow_alpha_spa,
    #                            alpha_spa, beta_spa, gamma_spa))

    # print("There are {} EEG features".format(features.size))

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
    
    # plot EEG signal and its PSD
    # freqs, power = welch(EEG_signal[:,0], fs=128., nperseg=128, scaling='density')
    # x = np.arange(0,1.0/128.0*EEG_signal.shape[0],1.0/128.0)

    # plt.figure(figsize=(20,10))

    # p1 = plt.subplot(211)
    # p2 = plt.subplot(212)
    # p1.plot(x,EEG_signal[:,0])
    # p1.set_ylabel("amplitude")
    # p1.set_xlabel("time(s)")
    # p1.set_title("EEG")

    # p2.plot(freqs,power)
    # p2.set_ylabel("PSD")
    # p2.set_xlabel("frequency(Hz)")
    # p2.set_title("EEG")

    # plt.show()

    return features


def ecg_preprocessing(signals):
    ''' Preprocessing for ECG signals '''
    ecg_all = ecg.ecg(signal=signals, sampling_rate=256., show=False)
    rpeaks = ecg_all['rpeaks'] # R-peak location indices.
    
    # ECG
    freqs, power = getfreqs_power(signals=signals,fs=256.)
    power_0_6 = []
    for i in range(60):
        power_0_6.append(getBand_Power(freqs,power,lower=0+(i * 0.1),upper=0.1+(i * 0.1)))

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
    per_above_IBI = float(IBI[IBI > mean_IBI + std_IBI].size)/float(IBI.size)
    per_below_IBI = float(IBI[IBI < mean_IBI - std_IBI].size)/float(IBI.size)
    
    # IBI
    freqs_, power_ = getfreqs_power(signals=IBI,fs=1.0/mean_IBI)
    power_001_008=getBand_Power(freqs_,power_,lower=0.01,upper=0.08)
    power_008_015=getBand_Power(freqs_,power_,lower=0.08,upper=0.15)
    power_015_050=getBand_Power(freqs_,power_,lower=0.15,upper=0.50)

    mean_heart_rate = np.mean(heart_rate)
    std_heart_rate = np.std(heart_rate)
    skew_heart_rate = skew(heart_rate)
    kurt_heart_rate = kurtosis(heart_rate)
    per_above_heart_rate = float(heart_rate[heart_rate >
                                      mean_heart_rate + std_heart_rate].size) / float(heart_rate.size)
    per_below_heart_rate = float(heart_rate[heart_rate <
                                      mean_heart_rate - std_heart_rate].size) / float(heart_rate.size)

    # features = np.concatenate(([rms_IBI, mean_IBI], power_0_6,
    #                            [power_001_008, power_008_015,
    #                             power_015_050, mean_heart_rate,
    #                             std_heart_rate, skew_heart_rate,
    #                             kurt_heart_rate, per_above_heart_rate,
    #                             per_below_heart_rate, std_IBI, skew_IBI,
    #                             kurt_IBI, per_above_IBI, per_below_IBI]))

    # print("There are {} ECG features".format(features.size))

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

    freqs, power = getfreqs_power(signals=signals,fs=128.)
    power_0_24 = []
    for i in range(21):
        power_0_24.append(getBand_Power(freqs,power,lower=0+(i*0.8/7),upper=0.1+(i*0.8/7)))
        
    SCSR = detrend(butter_lowpass_filter(nor_con_signals, 0.2, 128))
    SCVSR = detrend(butter_lowpass_filter(nor_con_signals, 0.08, 128))

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

    # features = np.concatenate(([mean, der_mean, neg_der_mean, neg_der_pro,
    #                             local_min, avg_rising_time], power_0_24,
    #                            [zcr_SCSR, zcr_SCVSR, mean_peak_SCSR, mean_peak_SCVSR]))

    # print("There are {} GSR features".format(features.size))

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
            gsr_signals = signals[:, -1]

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
    amigos_data = read_dataset('data')

    with open(os.path.join('data', 'features.p'), 'wb') as pickle_file:
        pickle.dump(amigos_data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    main()
    
