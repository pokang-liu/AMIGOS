#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multiscale Permutation Entropy Implementation
"""

from argparse import ArgumentParser
import itertools
import os
import time

from biosppy.signals import ecg
import numpy as np

from config import SUBJECT_NUM, VIDEO_NUM, SAMPLE_RATE, MISSING_DATA_SUBJECT
from utils import butter_highpass_filter


def coarse_graining(signal, scale):
    """Coarse-graining the signal.

    Arguments:
        signal: original signal,
        scale: desired scale
    Return:
        new_signal: coarse-grained signal
    """
    new_length = int(np.fix(len(signal) / scale))
    new_signal = np.zeros(new_length)
    for i in range(new_length):
        new_signal[i] = np.mean(signal[i * scale:(i + 1) * scale])

    return new_signal


def permutation_frequency(signal, emb_dim, delay):
    """ Calculate permutation frequency.

    Arguments:
        signal: input signal,
        emb_dim: embedding dimension,
        delay: time delay
    Return:
        prob: permutation frequency of the signal
    """
    length = len(signal)
    permutations = np.array(list(itertools.permutations(range(emb_dim))))
    count = np.zeros(len(permutations))
    for i in range(length - delay * (emb_dim - 1)):
        motif_index = np.argsort(signal[i:i + delay * emb_dim:delay])
        for k, perm in enumerate(permutations):
            if (perm == motif_index).all():
                count[k] += 1

    prob = count / sum(count)

    return prob


def multiscale_permutation_entropy(signal, scale, emb_dim, delay):
    """ Calculate multiscale permutation entropy.

    Arguments:
        signal: input signal,
        scale: coarse graining scale,
        emd_dim: embedding dimension,
        delay: time delay
    Return:
        mpe: multiscale permutation entropy value of the signal
    """
    cg_signal = coarse_graining(signal, scale)
    prob = permutation_frequency(cg_signal, emb_dim, delay)
    prob = list(filter(lambda p: p != 0., prob))
    mpe = -1 * np.sum(prob * np.log(prob))

    return mpe

def refined_composite_multiscale_permutation_entropy(signal, scale, emb_dim, delay):
    """ Calculate refined compositie multiscale permutation entropy.

    Arguments:
        signal: input signal,
        scale: coarse graining scale,
        emd_dim: embedding dimension,
        delay: time delay
    Return:
        rcmpe: refined compositie multiscale permutation entropy value of the signal
    """
    probs = []
    for i in range(scale):
        cg_signal = coarse_graining(signal, i + 1)
        tmp_prob = permutation_frequency(cg_signal, emb_dim, delay)
        probs.append(tmp_prob)
    prob = np.mean(probs, axis=0)
    prob = list(filter(lambda p: p != 0., prob))
    rcmpe = -1 * np.sum(prob * np.log(prob))

    return rcmpe


def multivariate_multiscale_permutation_entropy(signals, scale, emb_dim, delay):
    """ Calculate multivariate multiscale permutation entropy.

    Arguments:
        signals: input signals,
        scale: coarse graining scale,
        emb_dim: embedding dimension (m),
        delay: time delay
    Return:
        mvpe: multivariate permutation entropy value of the signal
    """
    num_channels = signals.shape[0]
    length = signals.shape[1]

    new_length = int(np.fix(length / scale))
    cg_signals = np.zeros((num_channels, new_length))
    for c in range(num_channels):
        cg_signals[c] = coarse_graining(signals[c], scale)

    permutations = np.array(list(itertools.permutations(range(emb_dim))))
    count = np.zeros((num_channels, len(permutations)))
    for i in range(num_channels):
        for j in range(new_length - delay * (emb_dim - 1)):
            motif_index = np.argsort(cg_signals[i, j:j + delay * emb_dim:delay])
            for k, perm in enumerate(permutations):
                if (perm == motif_index).all():
                    count[i, k] += 1

    count = [el for el in count.flatten() if el != 0]
    prob = np.divide(count, sum(count))
    mmpe = -sum(prob * np.log(prob))
    return mmpe


def eeg_preprocessing(signals):
    """ Preprocessing for EEG signals
    """
    print("EEG")
    signals = np.transpose(signals)
    length = signals.shape[1]
    tiled_mean = np.tile(signals.mean(1), (length, 1)).transpose()
    tiled_std = np.tile(signals.std(1), (length, 1)).transpose()
    nor_signals = (signals - tiled_mean) / tiled_std

    first_region = nor_signals.take([0, 13], 0)
    second_region = nor_signals.take([1, 2, 3, 10, 11, 12], 0)
    third_region = nor_signals.take([4, 9], 0)
    forth_region = nor_signals.take([3, 8], 0)
    fifth_region = nor_signals.take([6, 7], 0)

    eeg_mvmpe = []
    for s in range(1, 21):
        for d in range(2, 7):
            print("s{}, d{}".format(s, d), end='\r')
            eeg_mvmpe.append(multivariate_multiscale_permutation_entropy(first_region, s, d, 1))
            eeg_mvmpe.append(multivariate_multiscale_permutation_entropy(second_region, s, d, 1))
            eeg_mvmpe.append(multivariate_multiscale_permutation_entropy(third_region, s, d, 1))
            eeg_mvmpe.append(multivariate_multiscale_permutation_entropy(forth_region, s, d, 1))
            eeg_mvmpe.append(multivariate_multiscale_permutation_entropy(fifth_region, s, d, 1))

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
        for d in range(2, 7):
            print("s{}, d{}".format(s, d), end='\r')
            ibi_pe.append(refined_composite_multiscale_permutation_entropy(ibi, s, d, 1))

    return ibi_pe


def gsr_preprocessing(signal):
    """ Preprocessing for GSR signal
    """
    print("GSR")
    con_signal = 1.0 / signal
    nor_con_signal = (con_signal - np.mean(con_signal)) / np.std(con_signal)

    gsr_pe = []
    for s in range(1, 21):
        for d in range(2, 7):
            print("s{}, d{}".format(s, d), end='\r')
            gsr_pe.append(refined_composite_multiscale_permutation_entropy(nor_con_signal, s, d, 1))

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
    np.savetxt(os.path.join(args.data, 'gsr_rcmpe.csv'), amigos_data, delimiter=',')


if __name__ == '__main__':

    main()
