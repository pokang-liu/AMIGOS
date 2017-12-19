#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Functions for Preprocessing
'''

from argparse import ArgumentParser
import os
import warnings
import numpy as np
from biosppy.signals import ecg
from scipy.stats import skew, kurtosis

from utils import butter_highpass_filter, butter_lowpass_filter
from utils import getfreqs_power, getBand_Power, getFiveBands_Power
from utils import detrend
from utils import multiscale_entropy, permutation_entropy
from config import SUBJECT_NUM, VIDEO_NUM, SAMPLE_RATE, MISSING_DATA_SUBJECT

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


def eeg_preprocessing(signals):
    ''' Preprocessing for EEG signals '''
    # trans_signals = np.transpose(signals)

    features = []

    return features


def ecg_preprocessing(signals):
    ''' Preprocessing for ECG signals '''
    signals = butter_highpass_filter(signals, 1.0, 256.0)
    ecg_all = ecg.ecg(signal=signals, sampling_rate=256., show=False)
    rpeaks = ecg_all['rpeaks']  # R-peak location indices.

    ibi = np.array([])
    for i in range(len(rpeaks) - 1):
        ibi = np.append(ibi, (rpeaks[i + 1] - rpeaks[i]) / 128.0)

    # ibi_mse = list(multiscale_entropy(ibi, 10, None))
    ibi_pe = []
    for m in range(3, 4):
        ibi_pe.append(permutation_entropy(ibi, m, 1))

    features = ibi_pe

    return features


def gsr_preprocessing(signals):
    ''' Preprocessing for GSR signals '''
    nor_signals = (signals - np.mean(signals)) / np.std(signals)
    con_signals = 1.0 / signals
    nor_con_signals = (con_signals - np.mean(con_signals)) / np.std(con_signals)

    # nor_signals_mse = list(multiscale_entropy(nor_signals, 10, None))
    nor_signals_pe = []
    for m in range(3, 4):
        nor_signals_pe.append(permutation_entropy(nor_signals, m, 1))

    # nor_con_signals_mse = list(multiscale_entropy(nor_con_signals, 10, None))
    nor_con_signals_pe = []
    for m in range(3, 4):
        nor_con_signals_pe.append(permutation_entropy(nor_con_signals, m, 1))

    features = nor_signals_pe + nor_con_signals_pe

    return features


def read_dataset(path):
    ''' Read AMIGOS dataset '''
    amigos_data = np.array([])

    for sid in range(SUBJECT_NUM):
        for vid in range(VIDEO_NUM):
            if sid + 1 in MISSING_DATA_SUBJECT:
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

            features = np.array(eeg_features + ecg_features + gsr_features)

            amigos_data = np.vstack((amigos_data, features)) if amigos_data.size else features

    return amigos_data


def main():
    ''' Main function '''
    parser = ArgumentParser(
        description='Affective Computing with AMIGOS Dataset -- Feature Extraction')
    parser.add_argument('--data', type=str, default='./data')
    args = parser.parse_args()

    amigos_data = read_dataset(args.data)
    np.savetxt(os.path.join(args.data, 'entropy_features.csv'), amigos_data, delimiter=',')


if __name__ == '__main__':

    main()
