#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Affective Computing with AMIGOS Dataset
'''

import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from preprocessing import MISSING_DATA, SUBJECT_NUM

MISSING_DATA_IDX = []
for tup in MISSING_DATA:
    MISSING_DATA_IDX.append((tup[0] - 1) * 16 + tup[1] - 1)


def main():
    ''' Main function '''
    with open(os.path.join('data', 'features.p'), 'rb') as pickle_file:
        amigos_data = pickle.load(pickle_file)

    a_clf = SVC(kernel='linear')
    v_clf = SVC(kernel='linear')

    train_a_accuracy_history = []
    train_v_accuracy_history = []
    train_a_f1score_history = []
    train_v_f1score_history = []
    val_a_accuracy_history = []
    val_v_accuracy_history = []
    val_a_f1score_history = []
    val_v_f1score_history = []

    for i in range(SUBJECT_NUM):
        print("Leaving {} Subject Out".format(i + 1))
        train_data = np.array([])
        val_data = np.array([])

        for key, data_dict in amigos_data.items():
            tmp_array = np.array([])
            for _, item in data_dict.items():
                if isinstance(item, np.ndarray):
                    tmp_array = np.concatenate((tmp_array, item))
                else:
                    tmp_array = np.append(tmp_array, item)
            if key.split('_')[0] == str(i + 1):
                val_data = np.append(val_data, tmp_array)
            else:
                train_data = np.append(train_data, tmp_array)

        train_a_labels = []
        train_v_labels = []
        val_a_labels = []
        val_v_labels = []
        val_idx = np.arange(16) + i * 16

        with open(os.path.join('data', 'label.csv'), 'r') as label_file:
            for idx, line in enumerate(label_file.readlines()):
                if idx in MISSING_DATA_IDX:
                    continue
                if idx in val_idx:
                    val_a_labels.append(float(line.split(',')[0]))
                    val_v_labels.append(float(line.split(',')[1]))
                else:
                    train_a_labels.append(float(line.split(',')[0]))
                    train_v_labels.append(float(line.split(',')[1]))

        print('Training Arousal Model')
        a_clf.fit(train_data, train_a_labels)
        print('Training Valence Model')
        v_clf.fit(train_data, train_v_labels)

        train_a_predict_labels = a_clf.predict(train_data)
        train_v_predict_labels = v_clf.predict(train_data)

        val_a_predict_labels = a_clf.predict(val_data)
        val_v_predict_labels = v_clf.predict(val_data)

        train_a_accuracy = accuracy_score(train_a_labels, train_a_predict_labels)
        train_v_accuracy = accuracy_score(train_v_labels, train_v_predict_labels)
        train_a_f1score = f1_score(train_a_labels, train_a_predict_labels)
        train_v_f1score = f1_score(train_v_labels, train_v_predict_labels)

        val_a_accuracy = accuracy_score(val_a_labels, val_a_predict_labels)
        val_v_accuracy = accuracy_score(val_v_labels, val_v_predict_labels)
        val_a_f1score = f1_score(val_a_labels, val_a_predict_labels)
        val_v_f1score = f1_score(val_v_labels, val_v_predict_labels)

        print('Training Result')
        print("Arousal: Accuracy: {}, F1score: {}".format(train_a_accuracy, train_a_f1score))
        print("Valence: Accuracy: {}, F1score: {}".format(train_v_accuracy, train_v_f1score))

        print('Validating Result')
        print("Arousal: Accuracy: {}, F1score: {}".format(val_a_accuracy, val_a_f1score))
        print("Valence: Accuracy: {}, F1score: {}".format(val_v_accuracy, val_v_f1score))

        train_a_accuracy_history.append(train_a_accuracy)
        train_v_accuracy_history.append(train_v_accuracy)
        train_a_f1score_history.append(train_a_f1score)
        train_v_f1score_history.append(train_v_f1score)
        val_a_accuracy_history.append(val_a_accuracy)
        val_v_accuracy_history.append(val_v_accuracy)
        val_a_f1score_history.append(val_a_f1score)
        val_v_f1score_history.append(val_v_f1score)

        with open(os.path.join('model', "a_model_{}.p".format(i + 1)), 'wb') as pickle_file:
            pickle.dump(a_clf, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join('model', "v_model_{}.p".format(i + 1)), 'wb') as pickle_file:
            pickle.dump(v_clf, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    print('\nAverage Training Result')
    print("Arousal: Accuracy: {}, F1score: {}".format(
        np.mean(train_a_accuracy_history), np.mean(train_a_f1score_history)))
    print("Valence: Accuracy: {}, F1score: {}".format(
        np.mean(train_v_accuracy_history), np.mean(train_v_f1score_history)))

    print('Average Validating Result')
    print("Arousal: Accuracy: {}, F1score: {}".format(
        np.mean(val_a_accuracy_history), np.mean(val_a_f1score_history)))
    print("Valence: Accuracy: {}, F1score: {}".format(
        np.mean(val_v_accuracy_history), np.mean(val_v_f1score_history)))


if __name__ == '__main__':

    main()
