#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Plot Training History
'''

from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
HIS_DIR = os.path.join(BASE_DIR, 'history')
if not os.path.exists(HIS_DIR):
    os.makedirs(HIS_DIR)
PLT_DIR = os.path.join(BASE_DIR, 'plot')
if not os.path.exists(PLT_DIR):
    os.makedirs(PLT_DIR)


def plot_num_history(history):
    ''' Plot the history of accuracy when using different numbers of features '''
    path = os.path.join(PLT_DIR, history)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(HIS_DIR, history, 'train_a_history'), 'r') as train_a_file:
        train_a_acc = [float(a) for a in train_a_file.readlines()]
    with open(os.path.join(HIS_DIR, history, 'train_v_history'), 'r') as train_v_file:
        train_v_acc = [float(a) for a in train_v_file.readlines()]
    with open(os.path.join(HIS_DIR, history, 'val_a_history'), 'r') as val_a_file:
        val_a_acc = [float(a) for a in val_a_file.readlines()]
    with open(os.path.join(HIS_DIR, history, 'val_v_history'), 'r') as val_v_file:
        val_v_acc = [float(a) for a in val_v_file.readlines()]

    plt.plot(np.arange(len(train_a_acc)) + 1, train_a_acc, label='train', color='#428bca')
    plt.plot(np.arange(len(val_a_acc)) + 1, val_a_acc, label='valid', color='#d9534f')
    plt.legend()
    plt.grid()
    plt.title('Arousal Accuracy When Using Different number of features')
    plt.ylabel('Accurary')
    plt.xlabel('Number of features')
    plt.savefig(os.path.join(path, 'a_acc'))
    plt.close()

    plt.plot(np.arange(len(train_v_acc)) + 1, train_v_acc, label='train', color='#428bca')
    plt.plot(np.arange(len(val_v_acc)) + 1, val_v_acc, label='valid', color='#d9534f')
    plt.legend()
    plt.grid()
    plt.title('Valence Accuracy When Using Different number of features')
    plt.ylabel('Accurary')
    plt.xlabel('Number of features')
    plt.savefig(os.path.join(path, 'v_acc'))
    plt.close()


def plot_cost_history(history):
    ''' Plot the history of cost'''
    path = os.path.join(PLT_DIR, history)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(HIS_DIR, history, 'train_a_history'), 'r') as train_a_file:
        train_a_acc = [float(a) for a in train_a_file.readlines()]
    with open(os.path.join(HIS_DIR, history, 'train_v_history'), 'r') as train_v_file:
        train_v_acc = [float(a) for a in train_v_file.readlines()]
    with open(os.path.join(HIS_DIR, history, 'val_a_history'), 'r') as val_a_file:
        val_a_acc = [float(a) for a in val_a_file.readlines()]
    with open(os.path.join(HIS_DIR, history, 'val_v_history'), 'r') as val_v_file:
        val_v_acc = [float(a) for a in val_v_file.readlines()]

    plt.plot((np.arange(len(train_a_acc)) + 1) * 0.05, train_a_acc, label='train', color='#428bca')
    plt.plot((np.arange(len(val_a_acc)) + 1) * 0.05, val_a_acc, label='valid', color='#d9534f')
    plt.legend()
    plt.grid()
    plt.title('Arousal Accuracy on Different Cost')
    plt.ylabel('Accurary')
    plt.xlabel('Cost')
    plt.savefig(os.path.join(path, 'a_acc'))
    plt.close()

    plt.plot((np.arange(len(train_v_acc)) + 1) * 0.05, train_v_acc, label='train', color='#428bca')
    plt.plot((np.arange(len(val_v_acc)) + 1) * 0.05, val_v_acc, label='valid', color='#d9534f')
    plt.legend()
    plt.grid()
    plt.title('Valence Accuracy on Different Cost')
    plt.ylabel('Accurary')
    plt.xlabel('Cost')
    plt.savefig(os.path.join(path, 'v_acc'))
    plt.close()


def main():
    ''' Main function '''
    parser = ArgumentParser(description='Plot experiment result')
    parser.add_argument('--history', type=str, required=True)
    parser.add_argument('--type', type=str, choices=['num', 'cost'], required=True)
    args = parser.parse_args()

    if args.type == 'num':
        plot_num_history(args.history)
    if args.type == 'cost':
        plot_cost_history(args.history)

if __name__ == '__main__':

    main()
