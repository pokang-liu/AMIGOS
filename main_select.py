#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Affective Computing with AMIGOS Dataset
'''

from argparse import ArgumentParser
import os
import time
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

from config import MISSING_DATA_SUBJECT, SUBJECT_NUM, VIDEO_NUM, FEATURE_NAMES
from utils import fisher_idx


def main():
    ''' Main function '''
    parser = ArgumentParser(
        description='Affective Computing with AMIGOS Dataset -- Cross Validation')
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--feat', type=str, choices=['eeg', 'ecg', 'gsr', 'all'],
                        default='all', help='choose type of modality')
    parser.add_argument('--clf', type=str, choices=['gnb', 'svm', 'xgb'],
                        default='xgb', help='choose type of classifier')
    parser.add_argument('--nor', type=str, choices=['one', 'mean', 'no'],
                        default='one', help='choose type of classifier')
    parser.add_argument('--select', type=str, choices=['fisher', 'rfe', 'no'],
                        default='no', help='choose type of feature selection')
    parser.add_argument('--num', type=int, choices=range(0, 288),
                        default=20, help='choose the number of features')
    args = parser.parse_args()

    # read extracted features
    amigos_data = np.loadtxt(os.path.join(args.data, 'features.csv'), delimiter=',')

    # read labels and split to 0 and 1 based on individual mean
    labels = np.loadtxt(os.path.join(args.data, 'label.csv'), delimiter=',')
    labels = labels[:, :2]
    a_labels, v_labels = [], []
    for i in range(SUBJECT_NUM):
        if i + 1 in MISSING_DATA_SUBJECT:
            continue
        a_labels_mean = np.mean(labels[i * VIDEO_NUM:i * VIDEO_NUM + VIDEO_NUM, 0])
        v_labels_mean = np.mean(labels[i * VIDEO_NUM:i * VIDEO_NUM + VIDEO_NUM, 1])
        for idx, label in enumerate(labels[i * VIDEO_NUM:i * VIDEO_NUM + VIDEO_NUM, :]):
            a_tmp = 1 if label[0] > a_labels_mean else 0
            v_tmp = 1 if label[1] > v_labels_mean else 0
            a_labels.append(a_tmp)
            v_labels.append(v_tmp)
    a_labels, v_labels = np.array(a_labels), np.array(v_labels)

    # setup kfold cross validator
    kfold = KFold(n_splits=SUBJECT_NUM - len(MISSING_DATA_SUBJECT))

    # setup classifier
    if args.clf == 'gnb':
        a_clf = GaussianNB()
        v_clf = GaussianNB()
    elif args.clf == 'svm':
        a_clf = SVC(C=0.75, kernel='linear')
        v_clf = SVC(C=0.3, kernel='linear')
    elif args.clf == 'xgb':
        a_clf = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=1500,
            silent=True,
            objective="binary:logistic",
            nthread=-1,
            gamma=0,
            min_child_weight=1,
            max_delta_step=0,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            base_score=0.5,
            seed=0
        )
        v_clf = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=1000,
            silent=True,
            objective="binary:logistic",
            nthread=-1,
            gamma=0,
            min_child_weight=1,
            max_delta_step=0,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            base_score=0.5,
            seed=0
        )

    # setup feature selection
    if args.select == 'rfe':
        a_clf_select = RFE(a_clf, args.num, verbose=0)
        v_clf_select = RFE(v_clf, args.num, verbose=0)

    # initialize history list
    train_a_accuracy_history = []
    train_v_accuracy_history = []
    train_a_f1score_history = []
    train_v_f1score_history = []
    val_a_accuracy_history = []
    val_v_accuracy_history = []
    val_a_f1score_history = []
    val_v_f1score_history = []
    a_idx_history = np.zeros(amigos_data.shape[1])
    v_idx_history = np.zeros(amigos_data.shape[1])

    start_time = time.time()

    for idx, (train_idx, val_idx) in enumerate(kfold.split(amigos_data)):
        print(idx + 1, 'Fold Start')

        # collect data for cross validation
        if args.feat == 'eeg':
            train_data, val_data = amigos_data[train_idx][:175], amigos_data[val_idx][:175]
        elif args.feat == 'ecg':
            train_data, val_data = amigos_data[train_idx][175:256], amigos_data[val_idx][175:256]
        elif args.feat == 'gsr':
            train_data, val_data = amigos_data[train_idx][256:], amigos_data[val_idx][256:]
        elif args.feat == 'all':
            train_data, val_data = amigos_data[train_idx], amigos_data[val_idx]
        train_a_labels, val_a_labels = a_labels[train_idx], a_labels[val_idx]
        train_v_labels, val_v_labels = v_labels[train_idx], v_labels[val_idx]

        if args.nor == 'mean':
            # normalize using mean and std
            train_data_mean = np.mean(train_data, axis=0)
            train_data_std = np.std(train_data, axis=0)
            train_data = (train_data - train_data_mean) / train_data_std
            val_data_mean = np.mean(val_data, axis=0)
            val_data_std = np.std(val_data, axis=0)
            val_data = (val_data - val_data_mean) / val_data_std
        elif args.nor == 'one':
            # map features to [-1, 1]
            train_data_max = np.max(train_data, axis=0)
            train_data_min = np.min(train_data, axis=0)
            train_data = (train_data - train_data_min) / (train_data_max - train_data_min)
            train_data = train_data * 2 - 1
            val_data_max = np.max(val_data, axis=0)
            val_data_min = np.min(val_data, axis=0)
            val_data = (val_data - val_data_min) / (val_data_max - val_data_min)
            val_data = val_data * 2 - 1

        # fit feature selection
        if args.select == 'fisher':
            a_idx = fisher_idx(287, train_data, train_a_labels)
            v_idx = fisher_idx(287, train_data, train_v_labels)
            train_a_data, train_v_data = train_data[:, a_idx], train_data[:, v_idx]
            val_a_data, val_v_data = val_data[:, a_idx], val_data[:, v_idx]
        elif args.select == 'rfe':
            a_clf_select.fit(train_data, train_a_labels)
            v_clf_select.fit(train_data, train_v_labels)
            train_a_data = a_clf_select.transform(train_data)
            train_v_data = v_clf_select.transform(train_data)
            val_a_data = a_clf_select.transform(val_data)
            val_v_data = v_clf_select.transform(val_data)
            a_idx = np.where(a_clf_select.ranking_ == 1)
            v_idx = np.where(v_clf_select.ranking_ == 1)

        # record feature selection history
        for i in a_idx:
            a_idx_history[i] += 1
        for i in v_idx:
            v_idx_history[i] += 1

        # fit classifier
        a_clf.fit(train_a_data, train_a_labels)
        v_clf.fit(train_v_data, train_v_labels)

        # predict arousal and valence
        train_a_predict_labels = a_clf.predict(train_a_data)
        train_v_predict_labels = v_clf.predict(train_v_data)
        val_a_predict_labels = a_clf.predict(val_a_data)
        val_v_predict_labels = v_clf.predict(val_v_data)

        # metrics calculation (accuracy and f1 score)
        train_a_accuracy = accuracy_score(train_a_labels, train_a_predict_labels)
        train_v_accuracy = accuracy_score(train_v_labels, train_v_predict_labels)
        train_a_f1score = f1_score(train_a_labels, train_a_predict_labels, average='macro')
        train_v_f1score = f1_score(train_v_labels, train_v_predict_labels, average='macro')
        val_a_accuracy = accuracy_score(val_a_labels, val_a_predict_labels)
        val_v_accuracy = accuracy_score(val_v_labels, val_v_predict_labels)
        val_a_f1score = f1_score(val_a_labels, val_a_predict_labels, average='macro')
        val_v_f1score = f1_score(val_v_labels, val_v_predict_labels, average='macro')

        train_a_accuracy_history.append(train_a_accuracy)
        train_v_accuracy_history.append(train_v_accuracy)
        train_a_f1score_history.append(train_a_f1score)
        train_v_f1score_history.append(train_v_f1score)
        val_a_accuracy_history.append(val_a_accuracy)
        val_v_accuracy_history.append(val_v_accuracy)
        val_a_f1score_history.append(val_a_f1score)
        val_v_f1score_history.append(val_v_f1score)

        print('Training Result')
        print("Arousal: Accuracy: {:.4f}, F1score: {:.4f}".format(train_a_accuracy, train_a_f1score))
        print("Valence: Accuracy: {:.4f}, F1score: {:.4f}".format(train_v_accuracy, train_v_f1score))
        print('Validating Result')
        print("Arousal: Accuracy: {:.4f}, F1score: {:.4f}".format(val_a_accuracy, val_a_f1score))
        print("Valence: Accuracy: {:.4f}, F1score: {:.4f}".format(val_v_accuracy, val_v_f1score))

    print('\nDone. Duration: ', time.time() - start_time)

    print('\nAverage Training Result')
    print("Arousal => Accuracy: {:.4f}, F1score: {:.4f}".format(
        np.mean(train_a_accuracy_history), np.mean(train_a_f1score_history)))
    print("Valence => Accuracy: {:.4f}, F1score: {:.4f}".format(
        np.mean(train_v_accuracy_history), np.mean(train_v_f1score_history)))
    print('Average Validating Result')
    print("Arousal => Accuracy: {:.4f}, F1score: {:.4f}".format(
        np.mean(val_a_accuracy_history), np.mean(val_a_f1score_history)))
    print("Valence => Accuracy: {:.4f}, F1score: {:.4f}\n".format(
        np.mean(val_v_accuracy_history), np.mean(val_v_f1score_history)))

    # with open('train_a_history', 'a') as train_a_file:
    #     train_a_file.write("{}\n".format(np.mean(train_a_accuracy_history)))
    # with open('train_v_history', 'a') as train_v_file:
    #     train_v_file.write("{}\n".format(np.mean(train_v_accuracy_history)))
    # with open('val_a_history', 'a') as val_a_file:
    #     val_a_file.write("{}\n".format(np.mean(val_a_accuracy_history)))
    # with open('val_v_history', 'a') as val_v_file:
    #     val_v_file.write("{}\n".format(np.mean(val_v_accuracy_history)))

    sort_a_idx_history = np.argsort(a_idx_history)[::-1][:args.num]
    sort_v_idx_history = np.argsort(v_idx_history)[::-1][:args.num]

    a_feature_names = [FEATURE_NAMES[idx] for idx in sort_a_idx_history]
    v_feature_names = [FEATURE_NAMES[idx] for idx in sort_v_idx_history]

    print(a_feature_names)
    print(v_feature_names)

    # np.save("history/{}_{}_{}_a.npy".format(args.clf, args.select, args.num), sort_a_idx_history)
    # np.save("history/{}_{}_{}_v.npy".format(args.clf, args.select, args.num), sort_v_idx_history)


if __name__ == '__main__':

    main()
