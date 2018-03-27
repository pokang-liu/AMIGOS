#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Functions for Preprocessing
'''

from argparse import ArgumentParser
import os
import warnings
import numpy as np
from biosppy.signals import ecg
from scipy.stats import skew, kurtosis

from utils import butter_highpass_filter, butter_lowpass_filter
from utils import getfreqs_power, getBand_Power, getFiveBands_Power
from utils import detrend
from utils import multiscale_entropy, permutation_entropy,RC_composite_multiscale_entropy,RC_sample_entropy
from config import SUBJECT_NUM, VIDEO_NUM, SAMPLE_RATE, MISSING_DATA_SUBJECT

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


def eeg_preprocessing(signals):
    ''' Preprocessing for EEG signals '''
    # trans_signals = np.transpose(signals)
    column_0=signals[:,0]
    max_scale = 3
    max_m = 3
    feature_rcmse = np.zeros(( max_m,max_scale))
    feature_mse = np.zeros(( max_m,max_scale))
    for i in range(max_m):
        for j in range(max_scale):
            feature_rcmse[i,j]=RC_composite_multiscale_entropy(column_0,i+1,j+1,i+1, None)
            feature_mse[i,j]=multiscale_entropy(column_0,j+1,i+1, None)[-1]
    
    feature_rcmse = feature_rcmse.flatten()
    feature_mse = feature_mse.flatten()

    #ibi_mse = list(multiscale_entropy(ibi,3,3,3, None))
    #ibi_pe = []
    #for m in range(3, 4):
    #    ibi_pe.append(permutation_entropy(ibi, m, 1))
    
    features=np.array([])
    features=np.append(features,feature_mse)
    features=np.append(features,feature_rcmse)

    return features


def ecg_preprocessing(signals):
    ''' Preprocessing for ECG signals '''
    signals = butter_highpass_filter(signals, 1.0, 256.0)
    ecg_all = ecg.ecg(signal=signals, sampling_rate=256., show=False)
    rpeaks = ecg_all['rpeaks']  # R-peak location indices.

    ibi = np.array([])
    for i in range(len(rpeaks) - 1):
        ibi = np.append(ibi, (rpeaks[i + 1] - rpeaks[i]) / 128.0)
    #multiscale_entropy(time_series, scaling_factor , m , tolerance=None)
    #composite_multiscale_entropy(time_series, m, scale, tolerance=None)
    #
    #RC_composite_multiscale_entropy(time_series, sample_length, scale,m, tolerance=None):
    #  SAMPLE LENGTH need to equal m = =''
    #m is actually m+1 of the last ratio in numerator
    #rcmse_SCALE_M ######
    tol=0.2*np.std(ibi)
    '''
    max_scale = 3
    max_m = 3
    feature_rcmse = np.zeros(( max_m,max_scale))
    feature_mse = np.zeros(( max_m,max_scale))
    for i in range(max_m):
        for j in range(max_scale):
            feature_rcmse[i,j]=RC_composite_multiscale_entropy(ibi,i+1,j+1,i+1, tol)
            feature_mse[i,j]=multiscale_entropy(ibi,j+1,i+1,tol )[-1]
    
    feature_rcmse = feature_rcmse.flatten()
    feature_mse = feature_mse.flatten()
    '''
    #ibi_mse = list(multiscale_entropy(ibi,3,3,3, None))
    '''
    pe = []
    for m in range(2,8):
        pe.append(permutation_entropy(ibi, m, 1))
    '''
    #mse_2_3_0.1
    a= multiscale_entropy(ibi,2,3,0.1*np.std(ibi) )[-1]
    #rcmse_3_2_3_0.1
    b=RC_composite_multiscale_entropy(ibi,3,2,3, 0.1*np.std(ibi))
    #rcmse_2_2_2_0.2
    c=RC_composite_multiscale_entropy(ibi,2,2,2, 0.2*np.std(ibi))
    #rcmse_2_3_2_0.2
    d=RC_composite_multiscale_entropy(ibi,2,3,2, 0.2*np.std(ibi))
    #rcmse_1_2_1_0.1
    e=RC_composite_multiscale_entropy(ibi,1,2,1, 0.1*np.std(ibi))
    #mse_2_1_0.1
    f=multiscale_entropy(ibi,2,1,0.1*np.std(ibi) )[-1]
    features=np.array([])
    features=np.append(features,a)
    features=np.append(features,b)
    features=np.append(features,c)
    features=np.append(features,d)
    features=np.append(features,e)
    features=np.append(features,f)
    print(features.shape)
    
    return features


def gsr_preprocessing(signals):
    ''' Preprocessing for GSR signals '''
    nor_signals = (signals - np.mean(signals)) / np.std(signals)
    con_signals = 1.0 / signals
    nor_con_signals = (con_signals - np.mean(con_signals)) / np.std(con_signals)
    '''
    max_scale = 7
    max_m = 7
    feature_rcmse = np.zeros(( max_m,max_scale))
    feature_mse = np.zeros(( max_m,max_scale))
    for i in range(max_m):
        for j in range(max_scale):
            feature_rcmse[i,j]=RC_composite_multiscale_entropy(nor_con_signals,i+1,j+1,i+1, None)
            feature_mse[i,j]=multiscale_entropy(nor_con_signals,j+1,i+1, None)[-1]
    
    feature_rcmse = feature_rcmse.flatten()
    feature_mse = feature_mse.flatten()
    '''
    pe = []
    for m in range(3,6):
        pe.append(permutation_entropy(nor_con_signals, m, 1))

    features=np.array([])
    features=np.append(features,pe)
    #features=np.append(features,feature_rcmse)

    return features


def read_dataset(path):
    ''' Read AMIGOS dataset '''
    amigos_data = np.array([])

    for sid in range(SUBJECT_NUM):
        for vid in range(VIDEO_NUM):
            if sid + 1 in MISSING_DATA_SUBJECT:
                print("Skipping {}_{}.csv".format(sid + 1, vid + 1))
                continue
            print('Reading {}_{}.csv'.format(sid + 1, vid + 1))
            signals = np.genfromtxt(os.path.join(path, "{}_{}.csv".format(sid + 1, vid + 1)),
                                    delimiter=',')
            eeg_signals = signals[:, :14]
            ecg_signals = signals[:, 14]  # Column 14 or 15
            gsr_signals = signals[20:, -1]  # ignore the first 20 data, since there is noise in it

            #eeg_features = eeg_preprocessing(eeg_signals)
            ecg_features = ecg_preprocessing(ecg_signals)
            #gsr_features = gsr_preprocessing(gsr_signals)

            features = np.array( ecg_features )

            amigos_data = np.vstack((amigos_data, features)) if amigos_data.size else features

    return amigos_data


def main():
    ''' Main function '''
    parser = ArgumentParser(
        description='Affective Computing with AMIGOS Dataset -- Feature Extraction')
    parser.add_argument('--data', type=str, default='./data')
    args = parser.parse_args()

    amigos_data = read_dataset(args.data)
    np.savetxt(os.path.join(args.data, 'bestmse_features.csv'), amigos_data, delimiter=',')


if __name__ == '__main__':

    main()
