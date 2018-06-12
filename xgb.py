#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Affective Computing with AMIGOS Dataset (Tuning)
"""

from argparse import ArgumentParser
import os
import pickle
import warnings

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from tqdm import tqdm
import xgboost as xgb

from config import FEATURE_NAMES, MISSING_DATA_SUBJECT, SUBJECT_NUM
from utils import read_labels

warnings.filterwarnings("ignore")


def tuning(clf, param_name, tuning_params, data, labels, kf):
    """ Tuning one parameter
    """
    a_labels, v_labels = labels['a'], labels['v']
    a_acc_history = []
    v_acc_history = []

    for param in tqdm(tuning_params):
        # initialize history list
        train_a_accuracy_history = []
        train_v_accuracy_history = []
        val_a_accuracy_history = []
        val_v_accuracy_history = []

        # setup classifier
        a_clf, v_clf = clf['a'], clf['v']
        a_clf.set_params(**{param_name: param})
        v_clf.set_params(**{param_name: param})

        for train_idx, val_idx in kf.split(data):
            # collect data for cross validation
            train_data, val_data = data[train_idx], data[val_idx]
            train_a_labels, val_a_labels = a_labels[train_idx], a_labels[val_idx]
            train_v_labels, val_v_labels = v_labels[train_idx], v_labels[val_idx]

            # normalize using mean and std
            train_data_mean = np.mean(train_data, axis=0)
            train_data_std = np.std(train_data, axis=0)
            train_data = (train_data - train_data_mean) / train_data_std
            val_data_mean = np.mean(val_data, axis=0)
            val_data_std = np.std(val_data, axis=0)
            val_data = (val_data - val_data_mean) / val_data_std

            # fit classifier
            a_clf.fit(train_data, train_a_labels)
            v_clf.fit(train_data, train_v_labels)

            # predict arousal and valence
            train_a_predict_labels = a_clf.predict(train_data)
            train_v_predict_labels = v_clf.predict(train_data)
            val_a_predict_labels = a_clf.predict(val_data)
            val_v_predict_labels = v_clf.predict(val_data)

            # metrics calculation
            train_a_accuracy = f1_score(train_a_labels, train_a_predict_labels, average='macro')
            train_v_accuracy = f1_score(train_v_labels, train_v_predict_labels, average='macro')
            val_a_accuracy = f1_score(val_a_labels, val_a_predict_labels, average='macro')
            val_v_accuracy = f1_score(val_v_labels, val_v_predict_labels, average='macro')

            train_a_accuracy_history.append(train_a_accuracy)
            train_v_accuracy_history.append(train_v_accuracy)
            val_a_accuracy_history.append(val_a_accuracy)
            val_v_accuracy_history.append(val_v_accuracy)

        train_a_mean_accuracy = np.mean(train_a_accuracy_history)
        train_v_mean_accuracy = np.mean(train_v_accuracy_history)
        val_a_mean_accuracy = np.mean(val_a_accuracy_history)
        val_v_mean_accuracy = np.mean(val_v_accuracy_history)

        a_acc_history.append(val_a_mean_accuracy)
        v_acc_history.append(val_v_mean_accuracy)

    print('Tuning Result:')
    a_max_idx = np.argmax(a_acc_history)
    print("Arousal: Best value = {:.2f}, acc = {:.4f}".format(
        tuning_params[a_max_idx], a_acc_history[a_max_idx]))
    v_max_idx = np.argmax(v_acc_history)
    print("Valence: Best value = {:.2f}, acc = {:.4f}\n".format(
        tuning_params[v_max_idx], v_acc_history[v_max_idx]))

    return tuning_params[a_max_idx], tuning_params[v_max_idx], a_acc_history[a_max_idx], v_acc_history[v_max_idx]


def main():
    """ Main function
    """
    parser = ArgumentParser(
        description='Affective Computing with AMIGOS Dataset -- XGBoost')
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--feat', type=str, choices=['eeg', 'ecg', 'gsr', 'all'],
                        default='all', help='choose type of modality')
    parser.add_argument('--old', action='store_true')
    args = parser.parse_args()

    # read extracted features
    if args.old:
        data = np.loadtxt(os.path.join(args.data, 'features.csv'), delimiter=',')[:, :213]
    else:
        data = np.loadtxt(os.path.join(args.data, 'mse_mpe_features.csv'), delimiter=',')

    if args.feat == 'eeg':
        data = data[:, 108:213] if args.old else np.hstack(
            (data[:, 108:213], data[:, 213:233], data[:, 262:282]))
    elif args.feat == 'ecg':
        data = data[:, 32:108] if args.old else np.hstack(
            (data[:, 32:108], data[:, 233:242], data[:, 282:291]))
    elif args.feat == 'gsr':
        data = data[:, :32] if args.old else np.hstack(
            (data[:, :32], data[:, 242:262], data[:, 291:311]))

    # read labels and split to 0 and 1 by
    a_labels, v_labels = read_labels(os.path.join(args.data, 'label.csv'))

    # setup kfold cross validator
    sub_num = SUBJECT_NUM - len(MISSING_DATA_SUBJECT)
    kf = KFold(n_splits=sub_num)

    # classifier parameters
    with open(os.path.join(args.data, 'new_a_all_model.pkl'), 'rb') as f:
        a_param = pickle.load(f)
    with open(os.path.join(args.data, 'new_v_all_model.pkl'), 'rb') as f:
        v_param = pickle.load(f)

    a_clf = xgb.XGBClassifier(objective="binary:logistic")
    v_clf = xgb.XGBClassifier(objective="binary:logistic")
    a_clf.set_params(**a_param)
    v_clf.set_params(**v_param)

    train_a_accuracy_history = []
    train_v_accuracy_history = []
    val_a_accuracy_history = []
    val_v_accuracy_history = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print('Fold', fold + 1)
        # collect data for cross validation
        train_data, val_data = data[train_idx], data[val_idx]
        train_a_labels, val_a_labels = a_labels[train_idx], a_labels[val_idx]
        train_v_labels, val_v_labels = v_labels[train_idx], v_labels[val_idx]

        # normalize using mean and std
        train_data_mean = np.mean(train_data, axis=0)
        train_data_std = np.std(train_data, axis=0)
        train_data = (train_data - train_data_mean) / train_data_std
        val_data_mean = np.mean(val_data, axis=0)
        val_data_std = np.std(val_data, axis=0)
        val_data = (val_data - val_data_mean) / val_data_std

        # fit classifier
        a_clf.fit(train_data, train_a_labels)
        v_clf.fit(train_data, train_v_labels)

        # predict arousal and valence
        train_a_predict_labels = a_clf.predict(train_data)
        train_v_predict_labels = v_clf.predict(train_data)
        val_a_predict_labels = a_clf.predict(val_data)
        val_v_predict_labels = v_clf.predict(val_data)

        # metrics calculation
        train_a_accuracy = f1_score(train_a_labels, train_a_predict_labels, average='macro')
        train_v_accuracy = f1_score(train_v_labels, train_v_predict_labels, average='macro')
        val_a_accuracy = f1_score(val_a_labels, val_a_predict_labels, average='macro')
        val_v_accuracy = f1_score(val_v_labels, val_v_predict_labels, average='macro')

        train_a_accuracy_history.append(train_a_accuracy)
        train_v_accuracy_history.append(train_v_accuracy)
        val_a_accuracy_history.append(val_a_accuracy)
        val_v_accuracy_history.append(val_v_accuracy)

    print('Training Result:')
    print("Arousal: acc = {:.4f}".format(np.mean(train_a_accuracy_history)))
    print("Valence: acc = {:.4f}\n".format(np.mean(train_v_accuracy_history)))
    print('Validating Result:')
    print("Arousal: acc = {:.4f}".format(np.mean(val_a_accuracy_history)))
    print("Valence: acc = {:.4f}\n".format(np.mean(val_v_accuracy_history)))

    a_imp = a_clf.feature_importances_
    v_imp = v_clf.feature_importances_

    with open('a_imp', 'w') as f:
        for imp in a_imp:
            f.write("{}\n".format(imp))

    with open('v_imp', 'w') as f:
        for imp in v_imp:
            f.write("{}\n".format(imp))


if __name__ == '__main__':

    main()
