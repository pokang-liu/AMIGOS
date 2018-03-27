#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot Error Bar Plot for Entropy Features
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config import SUBJECT_NUM, MISSING_DATA_SUBJECT

NORM = 10


def main():
    """ Main function """
    # read extracted features
    amigos_data = np.loadtxt('data/pe_features.csv', delimiter=',')
    eeg_entropy = amigos_data[:, 287:302]
    ecg_entropy = amigos_data[:, 302:317]
    gsr_entropy = amigos_data[:, 317:332]

    # read labels and split to 0 and 1 by
    labels = np.loadtxt('data/label.csv', delimiter=',')
    labels = labels[:, :2]
    a_labels, v_labels = [], []
    for i in range(SUBJECT_NUM):
        if i + 1 in MISSING_DATA_SUBJECT:
            continue
        a_labels_mean = np.mean(labels[i * 16:i * 16 + 16, 0])
        v_labels_mean = np.mean(labels[i * 16:i * 16 + 16, 1])
        for idx, label in enumerate(labels[i * 16:i * 16 + 16, :]):
            a_tmp = 1 if label[0] > a_labels_mean else 0
            v_tmp = 1 if label[1] > v_labels_mean else 0
            a_labels.append(a_tmp)
            v_labels.append(v_tmp)
    a_labels, v_labels = np.array(a_labels), np.array(v_labels)

    pos_a_idx = np.where(a_labels == 1)[0]
    neg_a_idx = np.where(a_labels == 0)[0]
    pos_v_idx = np.where(v_labels == 1)[0]
    neg_v_idx = np.where(v_labels == 0)[0]

    pos_a_eeg_o4 = eeg_entropy[pos_a_idx, 2:13:5]
    neg_a_eeg_o4 = eeg_entropy[neg_a_idx, 2:13:5]
    pos_a_eeg_o5 = eeg_entropy[pos_a_idx, 3:14:5]
    neg_a_eeg_o5 = eeg_entropy[neg_a_idx, 3:14:5]
    pos_a_eeg_o6 = eeg_entropy[pos_a_idx, 4:15:5]
    neg_a_eeg_o6 = eeg_entropy[neg_a_idx, 4:15:5]
    pos_v_eeg_o4 = eeg_entropy[pos_v_idx, 2:13:5]
    neg_v_eeg_o4 = eeg_entropy[neg_v_idx, 2:13:5]
    pos_v_eeg_o5 = eeg_entropy[pos_v_idx, 3:14:5]
    neg_v_eeg_o5 = eeg_entropy[neg_v_idx, 3:14:5]
    pos_v_eeg_o6 = eeg_entropy[pos_v_idx, 4:15:5]
    neg_v_eeg_o6 = eeg_entropy[neg_v_idx, 4:15:5]

    pos_a_ecg_o4 = ecg_entropy[pos_a_idx, 2:13:5]
    neg_a_ecg_o4 = ecg_entropy[neg_a_idx, 2:13:5]
    pos_a_ecg_o5 = ecg_entropy[pos_a_idx, 3:14:5]
    neg_a_ecg_o5 = ecg_entropy[neg_a_idx, 3:14:5]
    pos_a_ecg_o6 = ecg_entropy[pos_a_idx, 4:15:5]
    neg_a_ecg_o6 = ecg_entropy[neg_a_idx, 4:15:5]
    pos_v_ecg_o4 = ecg_entropy[pos_v_idx, 2:13:5]
    neg_v_ecg_o4 = ecg_entropy[neg_v_idx, 2:13:5]
    pos_v_ecg_o5 = ecg_entropy[pos_v_idx, 3:14:5]
    neg_v_ecg_o5 = ecg_entropy[neg_v_idx, 3:14:5]
    pos_v_ecg_o6 = ecg_entropy[pos_v_idx, 4:15:5]
    neg_v_ecg_o6 = ecg_entropy[neg_v_idx, 4:15:5]

    pos_a_gsr_o4 = gsr_entropy[pos_a_idx, 2:13:5]
    neg_a_gsr_o4 = gsr_entropy[neg_a_idx, 2:13:5]
    pos_a_gsr_o5 = gsr_entropy[pos_a_idx, 3:14:5]
    neg_a_gsr_o5 = gsr_entropy[neg_a_idx, 3:14:5]
    pos_a_gsr_o6 = gsr_entropy[pos_a_idx, 4:15:5]
    neg_a_gsr_o6 = gsr_entropy[neg_a_idx, 4:15:5]
    pos_v_gsr_o4 = gsr_entropy[pos_v_idx, 2:13:5]
    neg_v_gsr_o4 = gsr_entropy[neg_v_idx, 2:13:5]
    pos_v_gsr_o5 = gsr_entropy[pos_v_idx, 3:14:5]
    neg_v_gsr_o5 = gsr_entropy[neg_v_idx, 3:14:5]
    pos_v_gsr_o6 = gsr_entropy[pos_v_idx, 4:15:5]
    neg_v_gsr_o6 = gsr_entropy[neg_v_idx, 4:15:5]

    # Arousal EEG PE
    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(10, 31, 10), np.mean(pos_a_eeg_o4, axis=0) / NORM, yerr=np.std(pos_a_eeg_o4) / pos_a_eeg_o4.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(10, 31, 10), np.mean(neg_a_eeg_o4, axis=0) / NORM, yerr=np.std(neg_a_eeg_o4) / neg_a_eeg_o4.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/a_eeg_o4.png')
    plt.close()

    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(10, 31, 10), np.mean(pos_a_eeg_o5, axis=0) / NORM, yerr=np.std(pos_a_eeg_o5) / pos_a_eeg_o5.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(10, 31, 10), np.mean(neg_a_eeg_o5, axis=0) / NORM, yerr=np.std(neg_a_eeg_o5) / neg_a_eeg_o5.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/a_eeg_o5.png')
    plt.close()

    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(10, 31, 10), np.mean(pos_a_eeg_o6, axis=0) / NORM, yerr=np.std(pos_a_eeg_o6) / pos_a_eeg_o6.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(10, 31, 10), np.mean(neg_a_eeg_o6, axis=0) / NORM, yerr=np.std(neg_a_eeg_o6) / neg_a_eeg_o6.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/a_eeg_o6.png')
    plt.close()

    # Valence EEG PE
    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(10, 31, 10), np.mean(pos_v_eeg_o4, axis=0) / NORM, yerr=np.std(pos_v_eeg_o4) / pos_v_eeg_o4.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(10, 31, 10), np.mean(neg_v_eeg_o4, axis=0) / NORM, yerr=np.std(neg_v_eeg_o4) / neg_v_eeg_o4.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/v_eeg_o4.png')
    plt.close()

    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(10, 31, 10), np.mean(pos_v_eeg_o5, axis=0) / NORM, yerr=np.std(pos_v_eeg_o5) / pos_v_eeg_o5.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(10, 31, 10), np.mean(neg_v_eeg_o5, axis=0) / NORM, yerr=np.std(neg_v_eeg_o5) / neg_v_eeg_o5.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/v_eeg_o5.png')
    plt.close()

    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(10, 31, 10), np.mean(pos_v_eeg_o6, axis=0) / NORM, yerr=np.std(pos_v_eeg_o6) / pos_v_eeg_o6.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(10, 31, 10), np.mean(neg_v_eeg_o6, axis=0) / NORM, yerr=np.std(neg_v_eeg_o6) / neg_v_eeg_o6.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/v_eeg_o6.png')
    plt.close()

    # Arousal ECG PE
    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(1, 4), np.mean(pos_a_ecg_o4, axis=0) / NORM, yerr=np.std(pos_a_ecg_o4) / pos_a_ecg_o4.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(1, 4), np.mean(neg_a_ecg_o4, axis=0) / NORM, yerr=np.std(neg_a_ecg_o4) / neg_a_ecg_o4.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/a_ecg_o4.png')
    plt.close()

    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(1, 4), np.mean(pos_a_ecg_o5, axis=0) / NORM, yerr=np.std(pos_a_ecg_o5) / pos_a_ecg_o5.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(1, 4), np.mean(neg_a_ecg_o5, axis=0) / NORM, yerr=np.std(neg_a_ecg_o5) / neg_a_ecg_o5.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/a_ecg_o5.png')
    plt.close()

    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(1, 4), np.mean(pos_a_ecg_o6, axis=0) / NORM, yerr=np.std(pos_a_ecg_o6) / pos_a_ecg_o6.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(1, 4), np.mean(neg_a_ecg_o6, axis=0) / NORM, yerr=np.std(neg_a_ecg_o6) / neg_a_ecg_o6.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/a_ecg_o6.png')
    plt.close()

    # Valence ECG PE
    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(1, 4), np.mean(pos_v_ecg_o4, axis=0) / NORM, yerr=np.std(pos_v_ecg_o4) / pos_v_ecg_o4.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(1, 4), np.mean(neg_v_ecg_o4, axis=0) / NORM, yerr=np.std(neg_v_ecg_o4) / neg_v_ecg_o4.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/v_ecg_o4.png')
    plt.close()

    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(1, 4), np.mean(pos_v_ecg_o5, axis=0) / NORM, yerr=np.std(pos_v_ecg_o5) / pos_v_ecg_o5.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(1, 4), np.mean(neg_v_ecg_o5, axis=0) / NORM, yerr=np.std(neg_v_ecg_o5) / neg_v_ecg_o5.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/v_ecg_o5.png')
    plt.close()

    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(1, 4), np.mean(pos_v_ecg_o6, axis=0) / NORM, yerr=np.std(pos_v_ecg_o6) / pos_v_ecg_o6.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(1, 4), np.mean(neg_v_ecg_o6, axis=0) / NORM, yerr=np.std(neg_v_ecg_o6) / neg_v_ecg_o6.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/v_ecg_o6.png')
    plt.close()

    # Arousal GSR PE
    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(10, 31, 10), np.mean(pos_a_gsr_o4, axis=0) / NORM, yerr=np.std(pos_a_gsr_o4) / pos_a_gsr_o4.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(10, 31, 10), np.mean(neg_a_gsr_o4, axis=0) / NORM, yerr=np.std(neg_a_gsr_o4) / neg_a_gsr_o4.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/a_gsr_o4.png')
    plt.close()

    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(10, 31, 10), np.mean(pos_a_gsr_o5, axis=0) / NORM, yerr=np.std(pos_a_gsr_o5) / pos_a_gsr_o5.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(10, 31, 10), np.mean(neg_a_gsr_o5, axis=0) / NORM, yerr=np.std(neg_a_gsr_o5) / neg_a_gsr_o5.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/a_gsr_o5.png')
    plt.close()

    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(10, 31, 10), np.mean(pos_a_gsr_o6, axis=0) / NORM, yerr=np.std(pos_a_gsr_o6) / pos_a_gsr_o6.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(10, 31, 10), np.mean(neg_a_gsr_o6, axis=0) / NORM, yerr=np.std(neg_a_gsr_o6) / neg_a_gsr_o6.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/a_gsr_o6.png')
    plt.close()

    # Valence GSR PE
    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(10, 31, 10), np.mean(pos_v_gsr_o4, axis=0) / NORM, yerr=np.std(pos_v_gsr_o4) / pos_v_gsr_o4.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(10, 31, 10), np.mean(neg_v_gsr_o4, axis=0) / NORM, yerr=np.std(neg_v_gsr_o4) / neg_v_gsr_o4.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/v_gsr_o4.png')
    plt.close()

    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(10, 31, 10), np.mean(pos_v_gsr_o5, axis=0) / NORM, yerr=np.std(pos_v_gsr_o5) / pos_v_gsr_o5.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(10, 31, 10), np.mean(neg_v_gsr_o5, axis=0) / NORM, yerr=np.std(neg_v_gsr_o5) / neg_v_gsr_o5.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/v_gsr_o5.png')
    plt.close()

    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.errorbar(np.arange(10, 31, 10), np.mean(pos_v_gsr_o6, axis=0) / NORM, yerr=np.std(pos_v_gsr_o6) / pos_v_gsr_o6.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
    ax.errorbar(np.arange(10, 31, 10), np.mean(neg_v_gsr_o6, axis=0) / NORM, yerr=np.std(neg_v_gsr_o6) / neg_v_gsr_o6.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
    ax.legend()
    ax.margins(0.05)
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig('plot/permutation_entropy/v_gsr_o6.png')
    plt.close()

if __name__ == '__main__':

    main()
