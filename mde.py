#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multiscale Dispersion Entropy Implementation
"""

import numpy as np
from scipy.stats import norm


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


def ncdf_mapping(signal):
    """Map the signal into y from 0 to 1 with NCDF.

    Arguments:
        signal: original signal
    Return:
        mapped_signal: mapped signal
    """
    length = len(signal)
    mean = np.mean(signal)
    std = np.std(signal)
    ncdf = norm(loc=mean, scale=std)
    mapped_signal = np.zeros(length)
    for i in range(length):
        mapped_signal[i] = ncdf.cdf(signal[i])
    return mapped_signal


def dispersion_frequency(signal, classes, emd_dim, delay):
    """ Calculate dispersion frequency.

    Arguments:
        signal: input signal,
        classes: number of classes,
        emd_dim: embedding dimension,
        delay: time delay
    Return:
        de: dispersion frequency of the signal
    """
    length = len(signal)
    mapped_signal = ncdf_mapping(signal)
    z_signal = np.round(classes * mapped_signal + 0.5)
    dispersion = np.zeros(classes ** emd_dim)
    for i in range(length - (emd_dim - 1) * delay):
        tmp_pattern = z_signal[i:i + emd_dim * delay:delay]
        pattern_index = 0
        for idx, c in enumerate(reversed(tmp_pattern)):
            pattern_index += ((c - 1) * (classes ** idx))

        dispersion[int(pattern_index)] += 1

    prob = dispersion / np.sum(dispersion)
    return prob


def dispersion_entropy(signal, classes, emd_dim, delay):
    """ Calculate dispersion entropy.

    Arguments:
        signal: input signal,
        classes: number of classes,
        emd_dim: embedding dimension,
        delay: time delay
    Return:
        de: dispersion entropy value of the signal
    """
    prob = dispersion_frequency(signal, classes, emd_dim, delay)
    prob = list(filter(lambda p: p != 0., prob))
    de = -1 * np.sum(prob * np.log(prob))
    return de


def multiscale_dispersion_entropy(signal, scale, classes, emd_dim, delay):
    """ Calculate multi-scale dispersion entropy.

    Arguments:
        signal: input signal,
        scale: coarse graining scale,
        classes: number of classes,
        emd_dim: embedding dimension,
        delay: time delay
    Return:
        mde: multi-scale dispersion entropy value of the signal
    """
    cg_signal = coarse_graining(signal, scale)
    prob = dispersion_frequency(cg_signal, classes, emd_dim, delay)
    prob = list(filter(lambda p: p != 0., prob))
    mde = -1 * np.sum(prob * np.log(prob))
    return mde


def refined_composite_dispersion_entropy(signal, scale, classes, emd_dim, delay):
    """ Calculate refined compositie multi-scale dispersion entropy.

    Arguments:
        signal: input signal,
        scale: coarse graining scale,
        classes: number of classes,
        emd_dim: embedding dimension,
        delay: time delay
    Return:
        rcmde: refined compositie multi-scale dispersion entropy value of the signal
    """
    probs = []
    for i in range(scale):
        cg_signal = coarse_graining(signal, i + 1)
        tmp_prob = dispersion_frequency(cg_signal, classes, emd_dim, delay)
        probs.append(tmp_prob)
    prob = np.mean(probs, axis=0)
    prob = list(filter(lambda p: p != 0., prob))
    rcmde = -1 * np.sum(prob * np.log(prob))
    return rcmde


test_signal = np.random.randn(10)
print(test_signal)

de = dispersion_entropy(test_signal, 3, 2, 1)
print(de)

mde = multiscale_dispersion_entropy(test_signal, 2, 3, 2, 1)
print(mde)

rcmde = refined_composite_dispersion_entropy(test_signal, 2, 3, 2, 1)
print(rcmde)
