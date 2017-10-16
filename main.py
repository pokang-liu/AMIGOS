#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Affective Computing with AMIGOS Dataset
'''

import os
import pickle

def main():
    ''' Main function '''
    with open(os.path.join('data', 'features.p'), 'rb') as pickle_file:
        amigos_data = pickle.load(pickle_file)

    for key, item in amigos_data[0].items():
        print("{}: {}".format(key, item))

if __name__ == '__main__':

    main()
