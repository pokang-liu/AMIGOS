#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Functions for Preprocessing (Permutation Entropy)
'''

from argparse import ArgumentParser
import itertools
import os
import warnings
import time
import numpy as np
from biosppy.signals import ecg

from utils import butter_highpass_filter
from utils import permutation_entropy, multiscale_permutation_entropy
from config import SUBJECT_NUM, VIDEO_NUM, SAMPLE_RATE, MISSING_DATA_SUBJECT

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

ORDER = [2, 3, 4, 5, 6]


def eeg_preprocessing(signals):
    ''' Preprocessing for EEG signals '''
    print("EEG")
    signals = np.transpose(signals)
    num_channel = signals.shape[0]
    length = signals.shape[1]
    delay = 1
    features = []
    for s in [10, 20, 30]:
        new_length = int(np.fix(length / s))
        granulate_signals = np.zeros((num_channel, new_length))
        for c in range(num_channel):
            for i in range(new_length):
                granulate_signals[c, i] = np.mean(signals[c, i * s: (i + 1) * s])
        for o in ORDER:
            permutations = np.array(list(itertools.permutations(range(o))))
            count = np.zeros((num_channel, len(permutations)))
            for i in range(num_channel):
                for j in range(new_length - delay * (o - 1)):
                    motif_index = np.argsort(granulate_signals[i, j:j + delay * o:delay])
                    for k in range(len(permutations)):
                        if (permutations[k] == motif_index).all():
                            count[i, k] += 1

            channel_motifs_dist = count / (num_channel * (new_length - delay * (o - 1)))
            motifs_dist = np.sum(channel_motifs_dist, axis=0)
            cross_channels_pe = -1 * np.sum(motifs_dist * np.ma.log2(motifs_dist).filled(0))

            # channels_pe = -1 * np.sum(num_channel * channel_motifs_dist *
            #                           np.ma.log2(num_channel * channel_motifs_dist).filled(0), axis=1)

            features.append(cross_channels_pe)
            # features.extend(channels_pe.tolist())

    return features


def ecg_preprocessing(signals):
    ''' Preprocessing for ECG signals '''
    # print("ECG")
    # signals = butter_highpass_filter(signals, 1.0, SAMPLE_RATE)
    # ecg_all = ecg.ecg(signal=signals, sampling_rate=SAMPLE_RATE, show=False)
    # rpeaks = ecg_all['rpeaks']  # R-peak location indices.

    # ibi = np.array([])
    # for i in range(len(rpeaks) - 1):
    #     ibi = np.append(ibi, (rpeaks[i + 1] - rpeaks[i]) / SAMPLE_RATE)

    # ibi_pe = []
    # for s in [1, 2, 3]:
    #     for o in ORDER:
    #         ibi_pe.append(multiscale_permutation_entropy(ibi, o, 1, s))

    # features = ibi_pe

    return []


def gsr_preprocessing(signals):
    ''' Preprocessing for GSR signals '''
    # print("GSR")
    # nor_signals = (signals - np.mean(signals)) / np.std(signals)

    # nor_signals_pe = []
    # for s in [10, 20, 30]:
    #     for o in ORDER:
    #         nor_signals_pe.append(multiscale_permutation_entropy(nor_signals, o, 1, s))

    # features = nor_signals_pe

    return []


def read_dataset(path):
    ''' Read AMIGOS dataset '''
    amigos_data = np.array([])

    for sid in range(SUBJECT_NUM):
        for vid in range(VIDEO_NUM):
            if sid + 1 in MISSING_DATA_SUBJECT:
                print("Skipping {}_{}.csv".format(sid + 1, vid + 1))
                continue
            start_time = time.time()
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
            print('Duration:', time.time() - start_time, 's')

    return amigos_data


def main():
    ''' Main function '''
    parser = ArgumentParser(
        description='Affective Computing with AMIGOS Dataset -- Feature Extraction')
    parser.add_argument('--data', type=str, default='./data')
    args = parser.parse_args()

    amigos_data = read_dataset(args.data)
    np.savetxt(os.path.join(args.data, 'entropy_mspe_features.csv'), amigos_data, delimiter=',')


if __name__ == '__main__':

    main()
