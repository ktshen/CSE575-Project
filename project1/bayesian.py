import numpy as np
from scipy.stats import norm


class NaiveBayes:
    def __init__(self, labels_num, features_num):
        self.labels_num = np.array(labels_num)
        self.features_num = np.array(features_num)
        # Store mean and variance for each features, because p(Xi | y) requires mean and variance for different features
        self.mean = np.zeros((labels_num, features_num), dtype=np.float)
        self.var = np.zeros((labels_num, features_num), dtype=np.float)
        self.priori = np.zeros(labels_num, dtype=np.float)

    def train(self, data, labels):
        N = data.shape[0]
        # Get digit 7 and digit 8 label amount
        each_label_amount = np.array([(labels == y).sum() for y in range(self.labels_num)], dtype=np.float)
        # Update prior of labels
        self.priori = each_label_amount / N;
        # Update Mean value of Gaussian
        for y in range(self.labels_num):
            sum = np.sum(data[n] if labels[n] == y else 0.0 for n in range(N))
            self.mean[y] = sum / each_label_amount[y]
        # Update Variance of Gaussian
        for y in range(self.labels_num):
            sum = np.sum((data[n] - self.mean[y])**2 if labels[n] == y else 0.0 for n in range(N))
            self.var[y] = sum / each_label_amount[y]

    def predict(self, data):
        results = [self.negative_log_likelihood(data, y) for y in range(self.labels_num)]
        return np.argmin(results)

    def negative_log_likelihood(self, x, y):
        log_priori_y = -np.log(self.priori[y])
        log_posterior_x_given_y = -np.sum([
            self.wrap_as_log_gaussian(x[f], self.mean[y][f], self.var[y][f]) for f in range(self.features_num)
        ])
        return log_priori_y + log_posterior_x_given_y

    def wrap_as_log_gaussian(self, x, mean, var):
        # Return Gaussian Distribution in log form
        return norm(mean, var).logpdf(x)

    def eval(self, test_X, test_y):
        predicted = np.array([self.predict(t) for t in test_X])
        correct_7 = 0
        correct_8 = 0
        for i in range(len(test_y)):
            if not predicted[i] == test_y[i]:
                continue
            if predicted[i] == 0:
                correct_7 += 1
            else:
                correct_8 += 1
        print(f"Accuracy for digit 7: {correct_7 / float((test_y==0).sum()) * 100.0} %")
        print(f"Accuracy for digit 8: {correct_8 / float((test_y==1).sum()) * 100.0} %")
        print(f"Overall Accuracy: {(correct_7+correct_8) / float(len(test_y)) * 100.0} %")
