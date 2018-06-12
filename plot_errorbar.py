#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot Error Bar Plot for Entropy Features
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils import read_labels

NORM = 10


def main():
    """ Main function """
    # read extracted features
    amigos_data = np.loadtxt('data/mde_features.csv', delimiter=',')
    eeg_entropy = amigos_data[:, :200]
    ecg_entropy = amigos_data[:, 200:206]
    gsr_entropy = amigos_data[:, 206:]

    # read labels and split to 0 and 1 by local mean
    a_labels, v_labels = read_labels('data/signals/label.csv')

    pos_a_idx = np.where(a_labels == 1)[0]
    neg_a_idx = np.where(a_labels == 0)[0]
    pos_v_idx = np.where(v_labels == 1)[0]
    neg_v_idx = np.where(v_labels == 0)[0]

    # EEG
    for d in range(2, 4):
        for r in range(1, 6):
            for typ in ['a', 'v']:
                start_idx = (d - 2) * 5 + (r - 1)
                end_idx = 20 * 2 * 5
                if typ == 'a':
                    pos_values = eeg_entropy[pos_a_idx, start_idx:end_idx:10]
                    neg_values = eeg_entropy[neg_a_idx, start_idx:end_idx:10]
                elif typ == 'v':
                    pos_values = eeg_entropy[pos_v_idx, start_idx:end_idx:10]
                    neg_values = eeg_entropy[neg_v_idx, start_idx:end_idx:10]
                _, ax = plt.subplots()
                for axis in [ax.xaxis, ax.yaxis]:
                    axis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax.errorbar(np.arange(1, 21), np.mean(pos_values, axis=0) / NORM, yerr=np.std(pos_values) / pos_values.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
                ax.errorbar(np.arange(1, 21), np.mean(neg_values, axis=0) / NORM, yerr=np.std(neg_values) / neg_values.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
                ax.legend()
                ax.margins(0.05)
                ax.axis('tight')
                plt.tight_layout()
                plt.savefig("plot/mde/eeg_mmde/{}_eeg_d{}_r{}.png".format(typ, d, r))
                plt.close()

    # ECG
    for d in range(2, 4):
        for typ in ['a', 'v']:
            start_idx = d - 2
            end_idx = 3 * 2
            if typ == 'a':
                pos_values = ecg_entropy[pos_a_idx, start_idx:end_idx:2]
                neg_values = ecg_entropy[neg_a_idx, start_idx:end_idx:2]
            elif typ == 'v':
                pos_values = ecg_entropy[pos_v_idx, start_idx:end_idx:2]
                neg_values = ecg_entropy[neg_v_idx, start_idx:end_idx:2]
            _, ax = plt.subplots()
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.errorbar(np.arange(1, 4), np.mean(pos_values, axis=0) / NORM, yerr=np.std(pos_values) / pos_values.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
            ax.errorbar(np.arange(1, 4), np.mean(neg_values, axis=0) / NORM, yerr=np.std(neg_values) / neg_values.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
            ax.legend()
            ax.margins(0.05)
            ax.axis('tight')
            plt.tight_layout()
            plt.savefig("plot/mde/ecg_rcmde/{}_ecg_d{}.png".format(typ, d))
            plt.close()

    # GSR
    for d in range(2, 4):
        for typ in ['a', 'v']:
            start_idx = d - 2
            end_idx = 20 * 2
            if typ == 'a':
                pos_values = gsr_entropy[pos_a_idx, start_idx:end_idx:2]
                neg_values = gsr_entropy[neg_a_idx, start_idx:end_idx:2]
            elif typ == 'v':
                pos_values = gsr_entropy[pos_v_idx, start_idx:end_idx:2]
                neg_values = gsr_entropy[neg_v_idx, start_idx:end_idx:2]
            _, ax = plt.subplots()
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.errorbar(np.arange(1, 21), np.mean(pos_values, axis=0) / NORM, yerr=np.std(pos_values) / pos_values.shape[0], marker='o', markersize=4, label='pos', color='#E76F51', capsize=4)
            ax.errorbar(np.arange(1, 21), np.mean(neg_values, axis=0) / NORM, yerr=np.std(neg_values) / neg_values.shape[0], marker='o', markersize=4, label='neg', color='#44A261', capsize=4)
            ax.legend()
            ax.margins(0.05)
            ax.axis('tight')
            plt.tight_layout()
            plt.savefig("plot/mde/gsr_rcmde/{}_gsr_d{}.png".format(typ, d))
            plt.close()

if __name__ == '__main__':

    main()
