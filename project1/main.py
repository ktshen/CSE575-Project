from scipy.io import loadmat
import numpy as np
import argparse

from logistic_regression import LogisticRegression
from bayesian import NaiveBayes

DATASET_FILE = 'mnist_data.mat'


dataset = loadmat(DATASET_FILE)

trX = dataset["trX"]
tsX = dataset["tsX"]
trY = dataset["trY"][0]
tsY = dataset["tsY"][0]

trX_new = np.array([[np.mean(i), np.std(i)] for i in trX])
tsX_new = np.array([[np.mean(i), np.std(i)] for i in tsX])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classifier to be used')
    parser.add_argument(
        "-c",
        "--classifier",
        metavar="classifier",
        type=str,
        help="Classifier to be used",
        required=True,
    )
    args = parser.parse_args()
    classifier_str = args.classifier.lower()

    if classifier_str == "nb":
        nb = NaiveBayes(len(set(trY)), trX_new.shape[-1])
        nb.train(trX_new, trY)
        nb.eval(tsX_new, tsY)

    else:
        lr = LogisticRegression()
        lr.fit(trX_new, trY)
        lr.eval(tsX_new, tsY)
