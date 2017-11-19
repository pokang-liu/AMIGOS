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
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

from config import MISSING_DATA_SUBJECT, SUBJECT_NUM, VIDEO_NUM

from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

def main():
    a_feature_history=np.array([])
    v_feature_history=np.array([])
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
    parser.add_argument('--select', type=str, choices=['rfe', 'sfs', 'sbs','no'],
                        default='no', help='choose type of feature selection')
    parser.add_argument('--num', type=int, choices=range(0,256),
                        default=20, help='choose the number of features')
    args = parser.parse_args()

    # read extracted features
    amigos_data = np.loadtxt(os.path.join(args.data, 'features.csv'), delimiter=',')

    # read labels and split to 0 and 1 by
    labels = np.loadtxt(os.path.join(args.data, 'label.csv'), delimiter=',')
    labels = labels[:, :2]
    a_labels, v_labels = [], []
    #############use one([-1,1]) to do selection #################
    amigos_data_max = np.max(amigos_data, axis=0)
    amigos_data_min = np.min(amigos_data, axis=0)
    amigos_data = (amigos_data - amigos_data_min) / (amigos_data_max - amigos_data_min)
    amigos_data = amigos_data * 2 - 1
    
    ########## process the labels first#################
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

    # setup kfold cross validator
    kf = KFold(n_splits=SUBJECT_NUM - len(MISSING_DATA_SUBJECT))

    # setup classifier
    if args.clf == 'gnb':
        a_clf = GaussianNB()
        v_clf = GaussianNB()
    elif args.clf == 'svm':
        a_clf = SVC(C=0.9, kernel='linear')
        v_clf = SVC(C=0.3, kernel='linear')
    elif args.clf == 'xgb':
        a_clf = xgb.XGBClassifier(
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
            colsample_bytree=0.5,
            colsample_bylevel=0.5,
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
            colsample_bytree=0.5,
            colsample_bylevel=0.5,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            base_score=0.5,
            seed=0
        )
        
        #################
    if args.select == 'rfe':
        a_clf_select = RFE(a_clf, args.num,verbose=1)
        v_clf_select = RFE(v_clf, args.num,verbose=1)
    elif args.select == 'sfs':
        a_clf_select  = SFS(a_clf, k_features=args.num, forward=True, floating=False,\
        verbose=2,scoring='accuracy',cv=10)
        v_clf_select  = SFS(v_clf, k_features=args.num, forward=True, floating=False,\
        verbose=2,scoring='accuracy',cv=10)
    elif args.select == 'sbs':
        a_clf_select  = SFS(a_clf, k_features=args.num, forward=False, floating=False,\
        verbose=2,scoring='accuracy',cv=10)
        v_clf_select  = SFS(v_clf, k_features=args.num, forward=False, floating=False,\
        verbose=2,scoring='accuracy',cv=10)
        
       
        ###################fit###############
        
    a_clf_select.fit(amigos_data,a_labels)
    v_clf_select.fit(amigos_data,v_labels)
    #################delete other feature ##########
    a_data = a_clf_select.transform(amigos_data)
    v_data = v_clf_select.transform(amigos_data)
 #####################print the ranking of the features after the selections~~~#############
    if args.select == 'sfs' or args.select == 'sfs':
        a_clf_select.k_feature_idx_=np.array(a_clf_select.k_feature_idx_)
        a_clf_select.k_feature_idx_=np.array(v_clf_select.k_feature_idx_)
        print('a_clf_select.k_feature_idx_{}'.format(args.num))
        print(a_clf_select.k_feature_idx_)
        print('v_clf_select.k_feature_idx_{}'.format(args.num))
        print(v_clf_select.k_feature_idx_)
       
        if args.select == 'sfs':
            np.save('a_sfs_select.ranking_{}'.format(args.num),a_clf_select.k_feature_idx_)
            np.save('v_rfe_select.ranking_{}'.format(args.num),v_clf_select.k_feature_idx_)
        if args.select == 'sbs':   
            np.save('a_sbs_select.ranking_{}'.format(args.num),a_clf_select.k_feature_idx_)
            np.save('v_sbs_select.ranking_{}'.format(args.num),v_clf_select.k_feature_idx_)
           

    elif args.select == 'rfe':
        print('a_clf.ranking_')
        
        print(a_clf_select.ranking_)
        a_clf_select.ranking_=np.where(a_clf_select.ranking_==1)
        np.save('a_rfe_select.k_feature_idx_',a_clf_select.ranking_)
        print('v_clf.ranking_')
        print(v_clf_select.ranking_)
        v_clf_select.ranking_=np.where(v_clf_select.ranking_==1)
        np.save('v_rfe_select.k_feature_idx_',v_clf_select.ranking_)
            ####################################

            
            
            
            #############################
       # initialize history list
    train_a_accuracy_history = []
    train_v_accuracy_history = []
    train_a_f1score_history = []
    train_v_f1score_history = []
    val_a_accuracy_history = []
    val_v_accuracy_history = []
    val_a_f1score_history = []
    val_v_f1score_history = []
    a_feature_history=np.array([])
    v_feature_history=np.array([])
    start_time = time.time()

    for idx, (train_idx, val_idx) in enumerate(kf.split(amigos_data)):
        print(idx + 1, 'Fold Start')

        # collect data for cross validation
        if args.feat == 'eeg':
            train_data, val_data = amigos_data[train_idx][:175], amigos_data[val_idx][:175]
        elif args.feat == 'ecg':
            train_data, val_data = amigos_data[train_idx][175:256], amigos_data[val_idx][175:256]
        elif args.feat == 'gsr':
            train_data, val_data = amigos_data[train_idx][256:], amigos_data[val_idx][256:]
        elif args.feat == 'all':
            train_a_data, val_a_data = a_data[train_idx], a_data[val_idx]
            train_v_data, val_v_data = v_data[train_idx], v_data[val_idx]
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
            '''
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
'''
 

        # fit classifier
        a_clf.fit(train_a_data, train_a_labels)
        v_clf.fit(train_v_data, train_v_labels)
###################################################
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
        print("Arousal: Accuracy: {}, F1score: {}".format(train_a_accuracy, train_a_f1score))
        print("Valence: Accuracy: {}, F1score: {}".format(train_v_accuracy, train_v_f1score))

        print('Validating Result')
        print("Arousal: Accuracy: {}, F1score: {}".format(val_a_accuracy, val_a_f1score))
        print("Valence: Accuracy: {}, F1score: {}".format(val_v_accuracy, val_v_f1score))
        
        

            
    print('\nDone. Duration: ', time.time() - start_time)

    print('\nAverage Training Result')
    print("Arousal => Accuracy: {}, F1score: {}".format(
        np.mean(train_a_accuracy_history), np.mean(train_a_f1score_history)))
    print("Valence => Accuracy: {}, F1score: {}".format(
        np.mean(train_v_accuracy_history), np.mean(train_v_f1score_history)))

    print('Average Validating Result')
    print("Arousal => Accuracy: {}, F1score: {}".format(
        np.mean(val_a_accuracy_history), np.mean(val_a_f1score_history)))
    print("Valence => Accuracy: {}, F1score: {}".format(
        np.mean(val_v_accuracy_history), np.mean(val_v_f1score_history)))
    #########################################
   
if __name__ == '__main__':

    main()
