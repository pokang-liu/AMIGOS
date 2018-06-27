#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multiscale Dispersion Entropy Implementation
"""

from argparse import ArgumentParser
import itertools
import os
import time

from biosppy.signals import ecg
import numpy as np
from scipy.special import comb
from scipy.stats import norm

from config import SUBJECT_NUM, VIDEO_NUM, SAMPLE_RATE, MISSING_DATA_SUBJECT
from utils import butter_highpass_filter


def coarse_graining(signal, scale):
    """Coarse-graining the signals.

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


def ncdf_mapping(signal):
    """Map the signal into y from 0 to 1 with NCDF.

    Arguments:
        signal: original signal
    Return:
        mapped_signal: mapped signal
    """
    length = len(signal)
    mean = np.mean(signal)
    std = np.std(signal) if np.std(signal) != 0 else 0.001
    ncdf = norm(loc=mean, scale=std)
    mapped_signal = np.zeros(length)
    for i in range(length):
        mapped_signal[i] = ncdf.cdf(signal[i])

    return mapped_signal


def dispersion_frequency(signal, classes, emb_dim, delay):
    """Calculate dispersion frequency.

    Arguments:
        signal: input signal,
        classes: number of classes,
        emb_dim: embedding dimension,
        delay: time delay
    Return:
        prob: dispersion frequency of the signal
    """
    length = len(signal)
    mapped_signal = ncdf_mapping(signal)
    z_signal = np.round(classes * mapped_signal + 0.5)
    dispersions = np.zeros(classes ** emb_dim)
    for i in range(length - (emb_dim - 1) * delay):
        tmp_pattern = z_signal[i:i + emb_dim * delay:delay]
        pattern_index = 0
        for idx, c in enumerate(reversed(tmp_pattern)):
            c = classes if c == (classes + 1) else c
            pattern_index += ((c - 1) * (classes ** idx))

        dispersions[int(pattern_index)] += 1

    prob = dispersions / sum(dispersions)

    return prob


def dispersion_entropy(signal, classes, emb_dim, delay):
    """Calculate dispersion entropy.

    Arguments:
        signal: input signal,
        classes: number of classes,
        emd_dim: embedding dimension,
        delay: time delay
    Return:
        de: dispersion entropy value of the signal
    """
    prob = dispersion_frequency(signal, classes, emb_dim, delay)
    prob = list(filter(lambda p: p != 0., prob))
    de = -1 * np.sum(prob * np.log(prob))

    return de


def multiscale_dispersion_entropy(signal, scale, classes, emb_dim, delay):
    """ Calculate multiscale dispersion entropy.

    Arguments:
        signal: input signal,
        scale: coarse graining scale,
        classes: number of classes,
        emd_dim: embedding dimension,
        delay: time delay
    Return:
        mde: multiscale dispersion entropy value of the signal
    """
    cg_signal = coarse_graining(signal, scale)
    prob = dispersion_frequency(cg_signal, classes, emb_dim, delay)
    prob = list(filter(lambda p: p != 0., prob))
    mde = -1 * np.sum(prob * np.log(prob))

    return mde


def refined_composite_multiscale_dispersion_entropy(signal, scale, classes, emb_dim, delay):
    """ Calculate refined compositie multiscale dispersion entropy.

    Arguments:
        signal: input signal,
        scale: coarse graining scale,
        classes: number of classes,
        emd_dim: embedding dimension,
        delay: time delay
    Return:
        rcmde: refined compositie multiscale dispersion entropy value of the signal
    """
    probs = []
    for i in range(scale):
        cg_signal = coarse_graining(signal, i + 1)
        tmp_prob = dispersion_frequency(cg_signal, classes, emb_dim, delay)
        probs.append(tmp_prob)
    prob = np.mean(probs, axis=0)
    prob = list(filter(lambda p: p != 0., prob))
    rcmde = -1 * np.sum(prob * np.log(prob))

    return rcmde


def multivariate_multiscale_dispersion_entropy(signals, scale, classes, emb_dim, delay):
    """ Calculate multivariate multiscale dispersion entropy.

    Arguments:
        signals: input signals,
        scale: coarse graining scale,
        classes: number of classes,
        emb_dim: embedding dimension,
        delay: time delay
    Return:
        mvmde: multivariate multiscale dispersion entropy value of the signal
    """
    num_channels = signals.shape[0]
    length = signals.shape[1]
    z_signals = np.zeros((num_channels, int(np.fix(length / scale))))
    for i, sc in enumerate(signals):
        cg_signals = coarse_graining(sc, scale)
        mapped_signals = ncdf_mapping(cg_signals)
        z_signals[i] = np.round(classes * mapped_signals + 0.5)

    dispersion = np.zeros(classes ** emb_dim)
    num_patterns = (length - (emb_dim - 1) * delay) * \
        comb(emb_dim * num_channels, emb_dim)
    for i in range(length - (emb_dim - 1) * delay):
        mv_z_signals = z_signals[:, i:i + emb_dim * delay:delay].flatten()
        for tmp_pattern in itertools.combinations(mv_z_signals, emb_dim):
            pattern_index = 0
            for idx, c in enumerate(reversed(tmp_pattern)):
                c = classes if c == (classes + 1) else c
                pattern_index += ((c - 1) * (classes ** idx))

            dispersion[int(pattern_index)] += 1

    prob = dispersion / num_patterns
    prob = list(filter(lambda p: p != 0., prob))
    mvmde = -1 * np.sum(prob * np.log(prob))

    return mvmde


def eeg_preprocessing(signals):
    ''' Preprocessing for EEG signals '''
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

    eeg_mvmde = []
    for s in range(1, 21):
        for d in range(2, 4):
            print("s{}, d{}".format(s, d), end='\r')
            eeg_mvmde.append(multivariate_multiscale_dispersion_entropy(
                first_region, s, 6, d, 1))
            eeg_mvmde.append(multivariate_multiscale_dispersion_entropy(
                second_region, s, 6, d, 1))
            eeg_mvmde.append(multivariate_multiscale_dispersion_entropy(
                third_region, s, 6, d, 1))
            eeg_mvmde.append(multivariate_multiscale_dispersion_entropy(
                forth_region, s, 6, d, 1))
            eeg_mvmde.append(multivariate_multiscale_dispersion_entropy(
                fifth_region, s, 6, d, 1))

    return eeg_mvmde


def ecg_preprocessing(signal):
    ''' Preprocessing for ECG signal '''
    print("ECG")
    signal = butter_highpass_filter(signal, 1.0, SAMPLE_RATE)
    ecg_all = ecg.ecg(signal=signal, sampling_rate=SAMPLE_RATE, show=False)
    rpeaks = ecg_all['rpeaks']  # R-peak location indices.
    ibi = np.array([])
    for i in range(len(rpeaks) - 1):
        ibi = np.append(ibi, (rpeaks[i + 1] - rpeaks[i]) / SAMPLE_RATE)

    ibi_mde = []
    for s in range(1, 4):
        for d in range(2, 4):
            print("s{}, d{}".format(s, d), end='\r')
            ibi_mde.append(
                refined_composite_multiscale_dispersion_entropy(ibi, s, 6, d, 1))

    return ibi_mde


def gsr_preprocessing(signal):
    ''' Preprocessing for GSR signal '''
    print("GSR")
    nor_signal = (signal - np.mean(signal)) / np.std(signal)

    gsr_mde = []
    for s in range(1, 21):
        for d in range(2, 4):
            print("s{}, d{}".format(s, d), end='\r')
            gsr_mde.append(refined_composite_multiscale_dispersion_entropy(
                nor_signal, s, 6, d, 1))

    return gsr_mde


def read_dataset(path):
    ''' Read AMIGOS dataset '''
    amigos_data = np.array([])

    for sid in range(30, 40):
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
            # ignore the first 20 data, since there is noise in it
            gsr_signal = signals[20:, -1]

            eeg_features = eeg_preprocessing(eeg_signals)
            ecg_features = ecg_preprocessing(ecg_signal)
            gsr_features = gsr_preprocessing(gsr_signal)

            features = np.array(eeg_features + ecg_features + gsr_features)

            amigos_data = np.vstack(
                (amigos_data, features)) if amigos_data.size else features
            print('Duration:', time.time() - start_time, 's')

    return amigos_data


def main():
    """ Main function
    """
    parser = ArgumentParser(
        description='Affective Computing with AMIGOS Dataset -- Feature Extraction')
    parser.add_argument('--data', type=str, default='./data')
    args = parser.parse_args()

    amigos_data = read_dataset(args.data)
    np.savetxt(os.path.join(args.data, 'gsr_rcmde_4.csv'),
               amigos_data, delimiter=',')


if __name__ == '__main__':

    main()
