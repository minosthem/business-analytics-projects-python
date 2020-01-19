from questions import svm
from questions import ann
from questions import dimention_reduction as dr
from os.path import join, exists
import os
import utils
import numpy as np


def main():
    """
    Main function. Uses the python files in questions packages
    :return:
    """
    # create output directory
    os.makedirs(utils.outpath, exist_ok=True)
    # load training and test data
    print("Loading data from csv")
    train, test = utils.load_data()
    # keep unique labels (the possible classes)
    labels = np.unique(train[0].values)
    # create dictionary with labels as keys and a tuple as value
    # the tuple contains a training and a test dataset containing items
    # labeled with the respective key-label
    print("Creating dictionary with datasets per label")
    data_per_label = utils.get_data_per_label(labels, train, test)
    # answer question 1
    print("Find most and least similar to 5 digit")
    svm.most_least_similar_five(labels, data_per_label)
    # answer questions 2 & 3 - prepare data for 4
    print("Executing majority votes")
    preds, labels = svm.majority_voting(test[0].values, test.values[:, 1:])
    # answer question 4
    print("Find best ANN with majority votes input")
    best_ann_path = join(utils.outpath, "ann_best_classifier")
    if exists(best_ann_path):
        best_nn = utils.load_pickle(best_ann_path)
        best_size = best_nn.hidden_layer_sizes[0]
        print("Best size for ANN {}".format(str(best_size)))
    else:
        best_nn, best_size = ann.run_nn(preds, labels)

    ann.run_encoders(train.values, test.values, best_nn)
    dr.dimention_reduction(train, test, best_nn)


if __name__ == '__main__':
    main()
