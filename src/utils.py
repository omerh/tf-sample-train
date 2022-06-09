import os
import tensorflow as tf
import numpy as np
import sys
import subprocess


def prepare_fs():
    data_dir = os.path.join(os.getcwd(), '../data')
    os.makedirs(data_dir, exist_ok=True)

    raw_dir = os.path.join(os.getcwd(), '../data/raw')
    os.makedirs(raw_dir, exist_ok=True)

    train_dir = os.path.join(os.getcwd(), '../data/train')
    os.makedirs(train_dir, exist_ok=True)

    test_dir = os.path.join(os.getcwd(), '../data/test')
    os.makedirs(test_dir, exist_ok=True)
    return data_dir, raw_dir, train_dir, test_dir


def load_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()

    # Set dataset colum names
    columns = ['CRIM',
               'ZN',
               'INDUS',
               'CHAS',
               'NOX',
               'RM',
               'AGE',
               'DIS',
               'RAD',
               'TAX',
               'PTRATIO',
               'B',
               'LSTAT']

    return (x_train, y_train), (x_test, y_test)


def save_dataset_local(raw_dir, x_train, y_train, x_test, y_test):
    np.save(os.path.join(raw_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(raw_dir, 'x_test.npy'), x_test)
    np.save(os.path.join(raw_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(raw_dir, 'y_test.npy'), y_test)


def get_train_data(train_dir):
    x_train = np.load(os.path.join(train_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(train_dir, 'y_train.npy'))
    print('x train', x_train.shape, 'y train', y_train.shape)

    return x_train, y_train


def get_test_data(test_dir):
    x_test = np.load(os.path.join(test_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(test_dir, 'y_test.npy'))
    print('x test', x_test.shape, 'y test', y_test.shape)

    return x_test, y_test


def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
