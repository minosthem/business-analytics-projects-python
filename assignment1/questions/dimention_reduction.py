from sklearn.decomposition import PCA
from sklearn import random_projection
from sklearn.metrics import accuracy_score


def dimention_reduction(train, test, best_nn):
    """
    Perform dimention reduction with PCA and Gaussian Random Projection
    The ANN from question 4 is used in order to evaluate the dimention reduction
    methods
    :param train: training dataset
    :param test:  test dataset
    :param best_nn: ANN from question 4 using the best performing hidden layer size
    """
    train_data, test_data = train.values[:, 1:], test.values[:, 1:]
    train_labels, test_labels = train.values[:, 0], test.values[:, 0]

    # PCA
    pca = PCA(random_state=1234)
    new_train_data = pca.fit_transform(train_data, train_labels)
    new_test_data = pca.transform(test_data)
    best_nn.fit(new_train_data, train_labels)
    preds_pca = best_nn.predict(new_test_data)
    acc_pca = accuracy_score(test_labels, preds_pca)
    print("PCA with ANN from q4 has accuracy {}".format(str(acc_pca)))

    # Random Projection
    rd = random_projection.GaussianRandomProjection(n_components=100, random_state=1234)
    new_train_data_rd = rd.fit_transform(train_data, train_labels)
    new_test_data_rd = rd.transform(test_data)
    best_nn.fit(new_train_data_rd, train_labels)
    preds_rd = best_nn.predict(new_test_data_rd)
    acc_rd = accuracy_score(test_labels, preds_rd)
    print("Random projection with ANN from q4 has accuracy {}".format(str(acc_rd)))