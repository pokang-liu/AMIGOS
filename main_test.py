#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Affective Computing with AMIGOS Dataset from Wang senior
'''

from argparse import ArgumentParser
from math import ceil
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

from ALL_preprocess import MISSING_DATA, SUBJECT_NUM, VIDEO_NUM
from fisher import fisher, feature_selection


def main():
    ''' Main function '''
    filename = 'f_213_CRE.csv'
    AMIGOS_data = np.genfromtxt(filename, delimiter=',') # 0~212 feature, 213 arousal, 214 valence
    feature = AMIGOS_data[:,:213]
    labels = AMIGOS_data[:,213:] # 1D
    
    for i in range(SUBJECT_NUM): 
        a_labels_mean = np.mean(labels[i*16:i*16+16,0])
        v_labels_mean = np.mean(labels[i*16:i*16+16,1])
        for idx, label in enumerate(labels[i*16:i*16+16,:]):
            labels[idx+i*16][0] = 1 if label[0] > a_labels_mean else 0
            labels[idx+i*16][1] = 1 if label[1] > v_labels_mean else 0  
    a_label = labels[:,0]
    v_label = labels[:,1]

    train_a_accuracy_history = []
    train_v_accuracy_history = []
    train_a_f1score_history = []
    train_v_f1score_history = []
    val_a_accuracy_history = []
    val_v_accuracy_history = []
    val_a_f1score_history = []
    val_v_f1score_history = []
    
    train_a_1_history = []
    train_v_1_hostory = []
    val_a_1_history = []
    val_v_1_history = []
    
    batch_size = 16
   
    for i in range(ceil(feature.shape[0]/16)):
        print("Leaving {} Batch Out".format(i + 1))  
        
        train_data = feature
        for j in range(batch_size):
            train_data = np.delete(train_data,i*batch_size,axis=0)
            
        val_data = feature[i*16:i*16+16,:]
        
        # map features to [-1, 1]
        train_data_max = np.max(train_data, axis=0)
        train_data_min = np.min(train_data, axis=0)
        train_data = (train_data - train_data_min) / (train_data_max - train_data_min)
        train_data = train_data * 2 - 1
        val_data_max = np.max(val_data, axis=0)
        val_data_min = np.min(val_data, axis=0)
        val_data = (val_data - val_data_min) / (val_data_max - val_data_min)
        val_data = val_data * 2 - 1

        # get labels for cross validation
        train_a_labels = a_label
        for j in range(batch_size):
            train_a_labels = np.delete(train_a_labels,i*batch_size)
        train_v_labels = v_label
        for j in range(batch_size):
            train_v_labels = np.delete(train_v_labels,i*batch_size)
            
        val_a_labels = a_label[i*16:i*16+16]
        val_v_labels = v_label[i*16:i*16+16]

        ###############################################
        a_estimator = SVC(C=0.9,kernel="linear")
        v_estimator = SVC(C=0.3,kernel="linear")
        a_clf = RFE(a_estimator, 20)
        v_clf = RFE(v_estimator, 21)
        #################################################
        print('Training Arousal Model')
        a_clf.fit(train_data, train_a_labels)
        print('Training Valence Model')
        v_clf.fit(train_data, train_v_labels)
        
######################################################
        train_a_predict_labels = a_clf.predict(train_data)
        train_v_predict_labels = v_clf.predict(train_data)

        val_a_predict_labels = a_clf.predict(val_data)
        val_v_predict_labels = v_clf.predict(val_data)
        
        
######################################################
        train_a_labels = np.array(train_a_labels) # list to array
        train_v_labels = np.array(train_v_labels)
        val_a_labels = np.array(val_a_labels)
        val_v_labels = np.array(val_v_labels)

        train_a_accuracy = accuracy_score(train_a_labels, train_a_predict_labels)
        train_v_accuracy = accuracy_score(train_v_labels, train_v_predict_labels)
        train_a_f1score = f1_score(train_a_labels, train_a_predict_labels, average='weighted')
        train_v_f1score = f1_score(train_v_labels, train_v_predict_labels, average='weighted')

        val_a_accuracy = accuracy_score(val_a_labels, val_a_predict_labels)
        val_v_accuracy = accuracy_score(val_v_labels, val_v_predict_labels)
        val_a_f1score = f1_score(val_a_labels, val_a_predict_labels, average='weighted')
        val_v_f1score = f1_score(val_v_labels, val_v_predict_labels, average='weighted')

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

    print('Average Training Result')
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
