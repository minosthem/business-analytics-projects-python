import os
import pickle
from os.path import join

import numpy as np
import pandas as pd

outpath = join(os.getcwd(), "questions", "output")
train_path = join(os.getcwd(), "datasets", "mnist_train.csv")
test_path = join(os.getcwd(), "datasets", "mnist_test.csv")
num_classes = 10

c_grid = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
gamma_grid = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
hidden_layer_sizes = [5, 10, 15, 20]
num_folds = 10


def to_one_hot(true_labels, possible_classes):
    """
    Create vector label for each item. Fill the vector with 1 in the index of the label
    and with zero the remaining indices
    :param true_labels: the true labels
    :param possible_classes: possible labels (unique)
    :return: numpy array with vectors for the label of each item
    """
    res = [[1 if i == possible_classes.index(d) else 0 for i in range(len(possible_classes))] for d in true_labels]
    return np.asarray(res)


def load_data():
    """
    Use pandas library to load data from csv as dataframes
    Loads two datasets: train and test
    :return:
    """
    train = pd.read_csv(train_path, header=None)
    test = pd.read_csv(test_path, header=None)
    return train, test


def load_pickle(path):
    """
    Load objects (e.g. classifiers) from files
    :param path: the path to the file
    :return: the loaded object
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def write_pickle(o, path):
    """
    Write an object to file with pickle library
    :param o: the object
    :param path: path to file
    """
    with open(path, "wb") as f:
        return pickle.dump(o, f)


def get_label_pair_id(l1, l2):
    """
    Create label pair id with the two provided labels
    :param l1: first label
    :param l2: second label
    :return: their id
    """
    return "_".join(map(str, sorted([l1, l2])))


def get_data_per_label(labels, train, test):
    """
    Gets a list of the possible labels and the training and test dataframes
    and creates a dictionary: each entry is related to a specific label and contains
    the respective training and test datasets
    :param labels: list of unique labels
    :param train: training dataset
    :param test: test dataset
    :return: dictionary with the two datasets separated by label
    """
    data_per_label = {}
    for lbl in labels:
        tr, te = train[train[0] == lbl], test[test[0] == lbl]
        data_per_label[lbl] = (tr, te)
    return data_per_label
