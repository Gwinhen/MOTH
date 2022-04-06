# coding: utf-8

import keras
import numpy as np
import pickle
import scipy.io

from keras.datasets import cifar10


def cifar10_data():
    num_classes = 10

    # load data 
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')

    # get validation set
    indices = np.arange(2000)
    x_val   = x_train[indices]
    y_val   = y_train[indices]
    x_train = np.delete(x_train, indices, axis=0)
    y_train = np.delete(y_train, indices, axis=0)

    print('train:\t', x_train.shape)
    print('val:\t',   x_val.shape)
    print('test:\t',  x_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test


def svhn_data():
    path = 'data/svhn'
    num_classes = 10

    # load data 
    train_data = scipy.io.loadmat(f'{path}/train_32x32.mat')
    test_data  = scipy.io.loadmat(f'{path}/test_32x32.mat')

    x_train = np.transpose(train_data['X'], axes=[3, 0, 1, 2])
    x_train = x_train.astype('float32') / 255.0
    y_train = train_data['y']
    y_train[y_train == 10] = 0
    y_train = keras.utils.to_categorical(y_train, num_classes)

    x_test  = np.transpose(test_data['X'],  axes=[3, 0, 1, 2])
    x_test  = x_test.astype('float32') / 255.0
    y_test  = test_data['y']
    y_test[y_test == 10] = 0
    y_test  = keras.utils.to_categorical(y_test, num_classes)

    # get validation set
    indices = np.arange(6000)
    x_val   = x_train[indices]
    y_val   = y_train[indices]
    x_train = np.delete(x_train, indices, axis=0)
    y_train = np.delete(y_train, indices, axis=0)

    print('train:\t', x_train.shape)
    print('val:\t',   x_val.shape)
    print('test:\t',  x_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test


def lisa_data():
    data_dir = 'data/lisa/'

    # load data 
    x_train = np.load(data_dir + 'train_x.npy')
    y_train = np.load(data_dir + 'train_y.npy')
    x_val   = np.load(data_dir + 'val_x.npy')
    y_val   = np.load(data_dir + 'val_y.npy')
    x_test  = np.load(data_dir + 'test_x.npy')
    y_test  = np.load(data_dir + 'test_y.npy')

    print('train:\t', x_train.shape, y_train.shape)
    print('val:\t',   x_val.shape,   y_val.shape)
    print('test:\t',  x_test.shape,  y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test


def gtsrb_data():
    num_classes = 43

    training_file = 'data/gtsrb/train.p'
    testing_file  = 'data/gtsrb/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    # load data 
    x_train, y_train = train['features'], train['labels']
    x_test,  y_test  = test['features'],  test['labels']
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32)  / 255.0
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.to_categorical(y_test,  num_classes)

    # get validation set
    indices = np.load('data/gtsrb/val_idx.npy')
    x_val   = x_train[indices]
    y_val   = y_train[indices]
    x_train = np.delete(x_train, indices, axis=0)
    y_train = np.delete(y_train, indices, axis=0)

    print('train:\t', x_train.shape)
    print('val:\t',   x_val.shape)
    print('test:\t',  x_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test
