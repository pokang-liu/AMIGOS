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
from utils import multiscale_permutation_entropy
from config import SUBJECT_NUM, VIDEO_NUM, SAMPLE_RATE, MISSING_DATA_SUBJECT

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


def multivariate_multiscale_permutation_entropy(signals, scale, emb_dim, delay):
    """ Calculate multivariate multiscale permutation entropy.

    Arguments:
        signals: input signals,
        scale: coarse graining scale,
        emb_dim: embedding dimension,
        delay: time delay
    Return:
        mvpe: multivariate permutation entropy value of the signal
    """
    num_channels = signals.shape[0]
    length = signals.shape[1]

    new_length = int(np.fix(length / scale))
    granulate_signals = np.zeros((num_channels, new_length))
    for c in range(num_channels):
        for i in range(new_length):
            granulate_signals[c, i] = np.mean(signals[c, i * scale: (i + 1) * scale])

    permutations = np.array(list(itertools.permutations(range(emb_dim))))
    count = np.zeros((num_channels, len(permutations)))
    for i in range(num_channels):
        for j in range(new_length - delay * (emb_dim - 1)):
            motif_index = np.argsort(granulate_signals[i, j:j + delay * emb_dim:delay])
            for k, perm in enumerate(permutations):
                if (perm == motif_index).all():
                    count[i, k] += 1

    channel_motifs_dist = count / (num_channels * (new_length - delay * (emb_dim - 1)))
    motifs_dist = np.sum(channel_motifs_dist, axis=0)
    mmpe = -1 * np.sum(motifs_dist * np.ma.log2(motifs_dist).filled(0))

    return mmpe


def eeg_preprocessing(signals):
    """ Preprocessing for EEG signals
    """
    print("EEG")
    signals = np.transpose(signals)
    tiled_mean = np.tile(signals.mean(1), (4, 1)).transpose()
    tiled_std = np.tile(signals.std(1), (4, 1)).transpose()
    nor_signals = (signals - tiled_mean) / tiled_std

    first_region = nor_signals.take([1, 14], 0)
    second_region = nor_signals.take([2, 3, 4, 11, 12, 13], 0)
    third_region = nor_signals.take([5, 10], 0)
    forth_region = nor_signals.take([4, 9], 0)
    fifth_region = nor_signals.take([7, 8], 0)

    eeg_mvmpe = []
    for s in range(1, 21):
        eeg_mvmpe.append(multivariate_multiscale_permutation_entropy(first_region, s, 5, 1))
        eeg_mvmpe.append(multivariate_multiscale_permutation_entropy(second_region, s, 5, 1))
        eeg_mvmpe.append(multivariate_multiscale_permutation_entropy(third_region, s, 5, 1))
        eeg_mvmpe.append(multivariate_multiscale_permutation_entropy(forth_region, s, 5, 1))
        eeg_mvmpe.append(multivariate_multiscale_permutation_entropy(fifth_region, s, 5, 1))

    return eeg_mvmpe


def ecg_preprocessing(signal):
    """ Preprocessing for ECG signal
    """
    print("ECG")
    signal = butter_highpass_filter(signal, 1.0, SAMPLE_RATE)
    ecg_all = ecg.ecg(signal=signal, sampling_rate=SAMPLE_RATE, show=False)
    rpeaks = ecg_all['rpeaks']  # R-peak location indices.
    ibi = np.array([])
    for i in range(len(rpeaks) - 1):
        ibi = np.append(ibi, (rpeaks[i + 1] - rpeaks[i]) / SAMPLE_RATE)

    ibi_pe = []
    for s in range(1, 4):
        ibi_pe.append(multiscale_permutation_entropy(ibi, 5, 1, s))

    return ibi_pe


def gsr_preprocessing(signal):
    """ Preprocessing for GSR signal
    """
    print("GSR")
    con_signal = 1.0 / signal
    nor_con_signal = (con_signal - np.mean(con_signal)) / np.std(con_signal)

    gsr_pe = []
    for s in range(1, 21):
        gsr_pe.append(multiscale_permutation_entropy(nor_con_signal, 5, 1, s))

    return gsr_pe


def read_dataset(path):
    """ Read AMIGOS dataset
    """
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
            ecg_signal = signals[:, 14]  # Column 14 or 15
            gsr_signal = signals[20:, -1]  # ignore the first 20 data, since there is noise in it

            eeg_features = eeg_preprocessing(eeg_signals)
            ecg_features = ecg_preprocessing(ecg_signal)
            gsr_features = gsr_preprocessing(gsr_signal)

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
    np.savetxt(os.path.join(args.data, 'mpe_features.csv'), amigos_data, delimiter=',')


if __name__ == '__main__':

    main()
