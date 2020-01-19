import itertools
import pickle
from os.path import join, exists
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit

import utils

svm_per_label_pair = {}
pair_evals = {}
pair_predictions = {}
test_per_label_pair = {}


def most_least_similar_five(labels, data_per_label):
    """
    This function is used to train SVM models, trying multiple C and sigma values.
    For each label pair, we keep the best performing model (classifier compared in a validation set)
    and then the best SVMs trained with the "5" labeled data are executed with the test dataset.
    Based on the accuracies, we define the most and least similar number to 5 from the respective label pairs
    :param labels: a list with unique labels
    :param data_per_label: a dictionary with the unique labels as keys and respective datasets
    (train and test) as values
    """
    eval_per_pair = {}

    # q1
    # create label combinations to run SVMs
    for lbl1, lbl2 in itertools.combinations(labels, 2):
        label_pair, tr, te = get_train_test_pair_datasets(data_per_label, lbl1, lbl2)
        test_per_label_pair[label_pair] = te
        np.random.seed(1234)
        # shuffle the data
        np.random.shuffle(tr)
        # create eval file name
        eval_path = join(utils.outpath, "eval_lbl{}_{}".format(lbl1, lbl2))
        # check if file already exists
        evals = check_for_existing_evals(eval_path)
        # if table/array is completed with the evaluations continue to find the best SVM
        if evals[evals >= 0].size == len(utils.c_grid) * len(utils.gamma_grid):
            print("Eval done for the config ", label_pair)
        else:
            # evaluations are not extracted - train SVM models for c / sigma values
            produce_all_svms_for_c_and_gamma(eval_path, evals, lbl1, lbl2, tr)

        # best configs
        pair_eval = get_best_svm(evals, label_pair, lbl1, lbl2)
        eval_per_pair[label_pair] = pair_eval

    find_most_least_similar_to_five()


# questions 2 & 3
def majority_voting(labels, test):
    """
    Run all the SVMs for the test dataset and for each item in the
    dataset get the majority vote. Then transform the predictions and
    the labels to 0 and 1
    :param labels: the test labels
    :param test: the test dataset
    :return: all predictions and labels with 0 1
    """
    all_preds = np.zeros((len(test), 0), np.float32)

    # votes and samples per digit
    labelset = sorted(set(labels))

    # votes per item
    votes = np.zeros((len(test), len(labelset)), np.int32)
    for lbl, svm in svm_per_label_pair.items():
        l1, l2 = map(int, lbl.split("_"))
        preds = svm.predict(test)
        oh_pred = np.expand_dims(np.asarray([0 if p == l1 else 1 for p in preds]), axis=1)
        all_preds = np.hstack((all_preds, oh_pred))

        for idx, pred in enumerate(preds):
            votes[idx, pred] += 1
    majorities = np.argmax(votes, axis=1)
    acc = accuracy_score(labels, majorities)
    print("Majority voting accuracy:", acc)
    for lbl in labelset:
        lbl_maj = [majorities[i] for i in range(len(majorities)) if labels[i] == lbl]
        acc = accuracy_score([lbl] * len(lbl_maj), lbl_maj)
        print("Label: {} : majority voting accuracy: {}".format(lbl, acc))

    oh_labels = utils.to_one_hot(labels, labelset)
    print("shape of one-hot labels & metafeatures:", oh_labels.shape, all_preds.shape)
    return all_preds, oh_labels


def find_most_least_similar_to_five():
    """
    Get all SVM models trained with the digit 5 and test them
    with the respective test set. Keep the best accuracy to find
    the least similar number and the worst to find the most similar
    digit to 5
    """
    fives = []
    for lbl_pair, svm in svm_per_label_pair.items():
        if "5" in lbl_pair:
            test = test_per_label_pair[lbl_pair]
            preds = svm.predict(test[:, 1:])
            acc = accuracy_score(test[:, 0], preds)
            fives.append((lbl_pair, acc))
    print(fives)
    worst = min(fives, key=lambda x: x[1])
    best = max(fives, key=lambda x: x[1])
    print("Most similar to 5: {}, with pair accuracy {} ".format(*worst))
    print("Least similar to 5: {}, with pair accuracy {} ".format(*best))
    print("Min / max pair evals in the entire classification set:", min(pair_evals.values()), max(pair_evals.values()))


def get_best_svm(evals, label_pair, lbl1, lbl2):
    """
    Based on the evaluations, we decide - for a given label pair - which
    trained SVM model has the best performance and we keep it
    :param evals: the numpy array with the evaluations
    :param label_pair: the unique id of the label pair
    :param lbl1: the first label
    :param lbl2: the second label
    :return:
    """
    pair_eval = np.max(evals)
    pair_evals[label_pair] = pair_eval
    best_c, best_g = np.unravel_index(np.argmax(evals), evals.shape)
    # store model
    model_path = join(utils.outpath,
                      "model_lbl{}_{}_c{}_g{}".format(lbl1, lbl2, utils.c_grid[best_c], utils.gamma_grid[best_g]))
    svm = utils.load_pickle(model_path)
    svm_per_label_pair[label_pair] = svm
    return pair_eval


def produce_all_svms_for_c_and_gamma(eval_path, evals, lbl1, lbl2, tr):
    """
    Uses the c / gamma lists in utils to produce different SVM models for a given label pair
    Then the model is evaluated and this evaluation is stored in evals
    :param eval_path: the file name
    :param evals: the numpy array with the valuations
    :param lbl1: the first label
    :param lbl2: the second label
    :param tr: training dataset for these two labels
    """
    stfs = StratifiedShuffleSplit(n_splits=1, test_size=2000, train_size=3000, random_state=1234)
    train_idx, validation_idx = list(stfs.split(tr, tr[:, 0]))[0]
    training = tr[train_idx,:]
    validation = tr[validation_idx,:]
    for g, gamma in enumerate(utils.gamma_grid):
        for c, C in enumerate(utils.c_grid):
            model_path = join(utils.outpath, "model_lbl{}_{}_c{}_g{}".format(lbl1, lbl2, C, gamma))
            if exists(model_path):
                # print("Loading existing questions with parameters c: {}, gamma: {} for labels {} {}".format(C, gamma,
                #                                                                                      lbl1,
                #                                                                                    lbl2))
                svm = utils.load_pickle(model_path)
            else:
                # print("Training questions with parameters c: {}, gamma: {} for labels {} {}".format(C, gamma, lbl1,
                #                                                                                   lbl2))
                svm = SVC(C=C, kernel='rbf', gamma=gamma)
                svm.fit(training[:, 1:], training[:, 0])
                with open(model_path, "wb") as f:
                    pickle.dump(svm, f)

            # evaluate
            if evals[c][g] < 0:
                # print("Evaluating model.")
                preds = svm.predict(validation[:, 1:])
                acc = accuracy_score(validation[:, 0], preds)
                evals[c][g] = acc
                utils.write_pickle(evals, eval_path)


def check_for_existing_evals(eval_path):
    """
    Checks if evaluation file exists, otherwise creates a numpy 2D array
    filled with ones.
    :param eval_path: the file name
    :return: the evaluations either loaded from file (using pickle) either the new numpy array
    """
    if exists(eval_path):
        evals = utils.load_pickle(eval_path)
    else:
        evals = np.ones((len(utils.c_grid), len(utils.gamma_grid))) * -1
    return evals


def get_train_test_pair_datasets(data_per_label, lbl1, lbl2):
    """
    Based on a label pair, we concatenate the two training and test dataset
    and create a unique key from the label pair
    :param data_per_label: dictionary with the data separated per label
    :param lbl1: the first label of the pair
    :param lbl2: the second label of the pair
    :return: a unique key of the two labels, the training and test datasets
    """
    label_pair = utils.get_label_pair_id(lbl1, lbl2)
    tr1, te1 = [x.values for x in data_per_label[lbl1]]
    tr2, te2 = [x.values for x in data_per_label[lbl2]]
    tr, te = np.vstack((tr1, tr2)), np.vstack((te1, te2))
    return label_pair, tr, te
