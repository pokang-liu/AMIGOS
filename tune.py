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

from config import MISSING_DATA_SUBJECT, SUBJECT_NUM
from utils import read_labels

warnings.filterwarnings("ignore")


def tuning(clf, param_name, tuning_params, data, labels, kf):
    """ Tuning one parameter
    """
    a_labels, v_labels = labels
    a_data, v_data = data
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

        for train_idx, val_idx in kf.split(a_data):
            # normalize using mean and std
            a_data = (a_data - np.mean(a_data, axis=0)) / np.std(a_data, axis=0)
            v_data = (v_data - np.mean(v_data, axis=0)) / np.std(v_data, axis=0)

            # collect data for cross validation
            a_train_data, a_val_data = a_data[train_idx], a_data[val_idx]
            v_train_data, v_val_data = v_data[train_idx], v_data[val_idx]
            train_a_labels, val_a_labels = a_labels[train_idx], a_labels[val_idx]
            train_v_labels, val_v_labels = v_labels[train_idx], v_labels[val_idx]

            # fit classifier
            a_clf.fit(a_train_data, train_a_labels)
            v_clf.fit(v_train_data, train_v_labels)

            # predict arousal and valence
            train_a_predict_labels = a_clf.predict(a_train_data)
            train_v_predict_labels = v_clf.predict(v_train_data)
            val_a_predict_labels = a_clf.predict(a_val_data)
            val_v_predict_labels = v_clf.predict(v_val_data)

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
        description='Affective Computing with AMIGOS Dataset -- Tuning')
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--feat', type=str, choices=['eeg', 'ecg', 'gsr', 'all'],
                        default='all', help='choose type of modality')
    parser.add_argument('--clf', type=str, choices=['gnb', 'svm', 'xgb'],
                        default='xgb', help='choose type of classifier')
    parser.add_argument('--old', action='store_true')
    args = parser.parse_args()

    # read extracted features
    if args.old:
        data = np.loadtxt(os.path.join(args.data, 'features.csv'), delimiter=',')[:, :213]
        if args.feat == 'eeg':
            data = data[:, 108:213]
        elif args.feat == 'ecg':
            data = data[:, 32:108]
        elif args.feat == 'gsr':
            data = data[:, 0:32]
        amigos_data = (data, data)
    else:
        a_data = np.loadtxt(os.path.join(args.data, 'a_features.csv'), delimiter=',')
        v_data = np.loadtxt(os.path.join(args.data, 'v_features.csv'), delimiter=',')
        if args.feat == 'eeg':
            a_data = np.hstack((a_data[:, 108:213], a_data[:, 229:232]))
            v_data = np.hstack((v_data[:, 108:213], v_data[:, 217:316]))
        elif args.feat == 'ecg':
            a_data = np.hstack((a_data[:, 32:108], a_data[:, 213:215], a_data[:, 232:234]))
            v_data = np.hstack((v_data[:, 32:108], v_data[:, 213:217], v_data[:, 316:324]))
        elif args.feat == 'gsr':
            a_data = np.hstack((a_data[:, 0:32], a_data[:, 215:229], a_data[:, 234:254]))
            v_data = v_data[:, 0:32]
        amigos_data = (a_data, v_data)

    # read labels and split to 0 and 1 by
    a_labels, v_labels = read_labels(os.path.join(args.data, 'label.csv'))
    labels = (a_labels, v_labels)

    # setup kfold cross validator
    sub_num = SUBJECT_NUM - len(MISSING_DATA_SUBJECT)
    kf = KFold(n_splits=sub_num)

    # tune classifier parameters
    grid_search_params = {
        'max_depth': np.arange(1, 6),
        'n_estimators': np.arange(1, 101)
    }
    other_tuning_params = {
        'learning_rate': np.arange(0.01, 0.41, 0.01),
        'gamma': np.arange(0, 10.1, 0.5),
        'min_child_weight': np.arange(0.80, 1.21, 0.01),
        'max_delta_step': np.arange(0, 2.05, 0.05),
        'subsample': np.arange(1.00, 0.59, -0.01),
        'colsample_bytree': np.arange(1.00, 0.09, -0.01),
        'colsample_bylevel': np.arange(1.00, 0.09, -0.01),
        'reg_alpha': np.arange(0, 2.05, 0.05),
        'reg_lambda': np.arange(0.50, 2.55, 0.05),
        'scale_pos_weight': np.arange(0.80, 1.21, 0.01),
        'base_score': np.arange(0.40, 0.61, 0.01),
        'seed': np.arange(0, 41)
    }

    # grid search tuning
    a_best_params = {
        'max_depth': 1,
        'n_estimators': 1
    }
    v_best_params = {
        'max_depth': 1,
        'n_estimators': 1
    }
    a_acc, v_acc = 0, 0

    print('Tuning max_depth and n_estimators')
    for param in grid_search_params['max_depth']:
        print('max_depth', param)
        clf = {
            'a': xgb.XGBClassifier(max_depth=param, objective="binary:logistic"),
            'v': xgb.XGBClassifier(max_depth=param, objective="binary:logistic")
        }
        tuning_params = grid_search_params['n_estimators']
        a_param, v_param, tmp_a_acc, tmp_v_acc = tuning(
            clf, 'n_estimators', tuning_params, amigos_data, labels, kf)
        if tmp_a_acc >= a_acc:
            a_best_params['max_depth'] = param
            a_best_params['n_estimators'] = a_param
            a_acc = tmp_a_acc
        if tmp_v_acc >= v_acc:
            v_best_params['max_depth'] = param
            v_best_params['n_estimators'] = v_param
            v_acc = tmp_v_acc

    # tune other parameters
    for param_name, tuning_params in other_tuning_params.items():
        print('Tuning', param_name)
        clf = {
            'a': xgb.XGBClassifier(objective="binary:logistic"),
            'v': xgb.XGBClassifier(objective="binary:logistic")
        }
        clf['a'].set_params(**a_best_params)
        clf['v'].set_params(**v_best_params)
        a_param, v_param, _, _ = tuning(
            clf, param_name, tuning_params, amigos_data, labels, kf)
        a_best_params[param_name] = a_param
        v_best_params[param_name] = v_param

    ver = 'old' if args.old else 'new'
    with open(os.path.join(args.data, "{}_a_{}_model.pkl".format(ver, args.feat)), 'wb') as f:
        pickle.dump(a_best_params, f)
    with open(os.path.join(args.data, "{}_v_{}_model.pkl".format(ver, args.feat)), 'wb') as f:
        pickle.dump(v_best_params, f)


if __name__ == '__main__':

    main()
