#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Calculate P-value
'''

import os
import pickle
import numpy as np
from sklearn.feature_selection import f_regression

from ALL_preprocess import MISSING_DATA, SUBJECT_NUM, VIDEO_NUM

MISSING_DATA_IDX = []
for tup in MISSING_DATA:
    MISSING_DATA_IDX.append((tup[0] - 1) * 16 + tup[1] - 1)


def main():
    ''' Main function '''
    with open(os.path.join('data', 'features.p'), 'rb') as pickle_file:
        amigos_data = pickle.load(pickle_file)

    data = np.array([])
    for _, data_dict in amigos_data.items():
        tmp_array = np.array([])
        for _, f_dict in data_dict.items():
            for _, item in f_dict.items():
                tmp_array = np.append(tmp_array, item)

        data = np.vstack((data, tmp_array)) if data.size else tmp_array

    a_labels = []
    v_labels = []
    with open(os.path.join('data', 'label.csv'), 'r') as label_file:
        for idx, line in enumerate(label_file.readlines()):
            if idx in MISSING_DATA_IDX:
                continue
            a_labels.append(float(line.split(',')[0]))
            v_labels.append(float(line.split(',')[1]))
            if idx == SUBJECT_NUM * VIDEO_NUM - 1:
                break

    a_labels_median = np.median(a_labels)
    v_labels_median = np.median(v_labels)
    for idx, label in enumerate(a_labels):
        a_labels[idx] = 1 if label > a_labels_median else 0
        v_labels[idx] = 1 if label > v_labels_median else 0

    _, a_pvalues = f_regression(data, a_labels)
    _, v_pvalues = f_regression(data, v_labels)

    print('Use Arousal Labels')
    print("Number of features (p < 0.05): {}".format(a_pvalues[a_pvalues < 0.05].size))
    print('Use Valence Labels')
    print("Number of features (p < 0.05): {}".format(v_pvalues[v_pvalues < 0.05].size))

if __name__ == '__main__':

    main()
