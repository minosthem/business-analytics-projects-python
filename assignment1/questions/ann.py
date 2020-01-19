from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from os.path import join, exists
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from PIL import Image
import torch.optim as optim
import copy
import pickle

import utils
import numpy as np


def run_nn(data, labels):
    """
    Train a ANN with the data produced by the majority votes from previous questions.
    ANN is tested with one hidden layer with different sizes and one output layer with 10
    nodes (the number of possible classes). We keep the best performing ANN
    :param data: predictions from majority votes transformed to 0 1
    :param labels: list with vectors for each item with 0 1 (1 for the true label of an item)
    :return: the best performning ANN and its size in the hidden layer
    """
    classifiers = {}
    accuracies = {}
    kfold = KFold(n_splits=utils.num_folds)
    indices = list(kfold.split(data, labels))
    for size in utils.hidden_layer_sizes:
        print("Examining hidden layer size:", size, "on", len(indices), "trainval folds")
        classifier_accuracies = []
        classifier = MLPClassifier(hidden_layer_sizes=(size, utils.num_classes), max_iter=1000)
        classifiers[size] = classifier
        for train_idx, test_idx in indices:
            train_data, test_data = data[train_idx], data[test_idx]
            train_labels, test_labels = labels[train_idx], labels[test_idx]
            classifier.fit(train_data, train_labels)
            predictions = classifier.predict(test_data)
            acc = accuracy_score(test_labels, predictions)
            classifier_accuracies.append(acc)
        print("Running mean of data:", classifier_accuracies)
        avg_acc_class = np.mean(classifier_accuracies)
        accuracies[size] = avg_acc_class
        utils.write_pickle(classifier, join(utils.outpath, "ml_classifier_size_{}".format(str(size))))

    maximum = max(accuracies, key=accuracies.get)
    print("Best size for ANN {}".format(str(maximum)))
    print("Accuracy for the best performing ANN {}".format(str(accuracies[maximum])))
    utils.write_pickle(classifiers[maximum], join(utils.outpath, "ann_best_classifier"))
    return classifiers[maximum], maximum


def run_encoders(train, test, best_nn):
    """
    Creates encoders using PyTorch framework. The first encoder is for hidden layer with size 200
    and the second encoder is for the hidden layer 45
    :param train:  training dataset
    :param test:  test dataset
    :param best_nn:  ANN from question 4 with the best performing hidden layer size
    """
    print("Running encoders")
    # get training and test data & labels
    tr, te = np.ndarray.astype(train[:, 1:], np.float32), np.ndarray.astype(test[:, 1:], np.float32)
    trl, tel = train[:, 0], test[:, 0]

    # run the ANN from question 4 with the Input layer 784 to find the accuracy
    best_nn.fit(tr, trl)
    pred = best_nn.predict(te)
    acc = accuracy_score(tel, pred)
    print("Accuracy for Input Layer 784 is {}".format(str(acc)))

    print("Running encoders to 200")
    # train encoders
    if exists(join(utils.outpath, "encoder1.model")):
        # load encoder model if exists
        encoder1 = utils.load_pickle(join(utils.outpath, "encoder1.model"))
    else:
        # train a new encoder
        encoder1 = DeepNN(tr.shape[-1], 200)
        encoder1 = train_pytorch_model(encoder1, tr, trl, num_epochs=1000)
        # write encoder to file
        with open(join(utils.outpath, "encoder1.model"), "wb") as f:
            pickle.dump(encoder1, f)

    # get encodings to be used as input to the next layer
    tr200, te200 = encoder1.get_encodings(tr).detach().numpy(), encoder1.get_encodings(te).detach().numpy()

    # give the encodings as input to the ANN from question 4 to find the accuracy of this part
    best_nn.fit(tr200, trl)
    pred200 = best_nn.predict(te200)
    acc_200 = accuracy_score(tel, pred200)
    print("Accuracy for Hidden Layer 200 is {}".format(str(acc_200)))

    # plot images with the weights of each node of the hidden layer
    plot_weights_hidden_layer(encoder1, "hidden_layer_200_img_", (28, 28))

    print("Running encoders to 45")
    # second encoder
    if exists(join(utils.outpath, "encoder2.model")):
        # if exists load the encoder from file
        encoder2 = utils.load_pickle(join(utils.outpath, "encoder2.model"))
    else:
        # train new encoder for hidden layer 45
        encoder2 = DeepNN(tr200.shape[-1], 45)
        encoder2 = train_pytorch_model(encoder2, tr200, trl, num_epochs=1000)
        # write encoder to file
        with open(join(utils.outpath, "encoder2.model"), "wb") as f:
            pickle.dump(encoder2, f)
    # get the encodings from the encoder
    tr45, te45 = encoder2.get_encodings(tr200).detach().numpy(), encoder2.get_encodings(te200).detach().numpy()

    # run encoder output on the MLP from question 4
    best_nn.fit(tr45, trl)
    pred_45 = best_nn.predict(te45)
    acc_45 = accuracy_score(tel, pred_45)
    print("Accuracy for Hidden Layer 45 is {}".format(str(acc_45)))

    # plot the weights of each node of the hidden layer 45 into images
    plot_weights_hidden_layer(encoder2, "hidden_layer_45_img_", (20, 10))


class DeepNN(nn.Module):
    """
    Represents a DNN. In the constructor the fields
    linear_in, sigmoid and linear_out are initialized.
    Need to implement forward method
    """
    def __init__(self, input_dim, encoding_dim):
        super(DeepNN, self).__init__()
        self.linear_in = nn.Linear(input_dim, encoding_dim)
        self.sigmoid = nn.LogSigmoid()
        self.linear_out = nn.Linear(encoding_dim, input_dim)

    def forward(self, x):
        return self.sigmoid(self.linear_out(self.get_encodings(x)))

    def get_encodings(self, x):
        """
        Method to get the encodings to be provided to the next
        layer
        :param x: tensor from the encoder
        :return: the encodings
        """
        if type(x) != Tensor:
            x = Tensor(x)
        return self.sigmoid(self.linear_in(x))


class Dset:
    """
    Represents a dataset
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx], 'length': len(self.x[idx])}


def train_pytorch_model(model, data, labels, num_epochs=100):

    # create a training Dset
    trainset = Dset(data, labels)
    # create a DataLoader object
    dl = DataLoader(trainset, batch_size=100, shuffle=True)
    # optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    lossFunc = nn.MSELoss()
    best, bestloss = None, 99999999
    # train the network (epoch = how many times to train the network with the data)
    for epoch in range(num_epochs):
        for i, d in enumerate(dl):
            optimizer.zero_grad()
            out = model.forward(d['x'])
            loss = lossFunc(out, d['x'])
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0 and epoch > 0:
                print("Epoch {}, batch {}, loss {:.3f}".format(epoch, i, loss.item()))
            # save best model
            if bestloss > loss.item():
                best = copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model


def plot_weights_hidden_layer(encoder, basefilename, shp):
    """
    Given the encoder, the file name and the image size to be produced,
    the method iterates the nodes of a hidden layer and plots the all
    the weights
    :param encoder: the encoder from which we get the parameters
    :param basefilename:  the base filename extended with the ith node
    :param shp: the image size
    """
    trained_weights = (list(encoder.parameters())[0]).data.numpy()
    trained_weights -= np.min(trained_weights)
    trained_weights /= np.max(trained_weights)
    trained_weights = np.round(255*trained_weights)
    for i in range(len(trained_weights)):
        datum = trained_weights[i, :]
        if len(datum.shape) == 1:
            datum = np.expand_dims(datum, 1)
        reshaped_weight = np.ndarray.astype(np.reshape(datum, shp), np.uint8)
        img = Image.fromarray(reshaped_weight, 'L')
        img.save(join(utils.outpath, basefilename + str(i) + ".png"))
