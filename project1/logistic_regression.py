import numpy as np


class LogisticRegression:
    """
        This class is a basic form Logistic Regression
    """
    def __init__(self, learning_rate=0.01, epoches=1000000):
        # The more the epoches and the lower the learning, the better the result will be
        self.epoches = epoches
        self.learning_rate = learning_rate

    def fit(self, X, y):
        # Initialize parameters of weight
        self.theta = np.zeros(X.shape[1])

        for i in range(self.epoches):
            # Inner product between x and weight
            z = np.dot(X, self.theta)
            # classify the result
            h = self.sigmoid(z)
            # Update the weight by using gradient descent
            gradient = np.dot(X.T, (h-y))/y.size
            self.theta = self.theta - self.learning_rate * gradient

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def loss(self, h, y):
        # The main goal is to minimize this function in order to get the better result
        # J(θ) = (-yT * log(h) - (1-y)T * log(1-h))
        return (-(1-y) * np.log(1 - h) + (-y * np.log(h))).mean()

    def predict_probability(self, X):
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold):
        return self.predict_probability(X) >= threshold

    def eval(self, test_X, test_Y):
        predicted = self.predict(test_X, 0.52)
        correct = 0
        for i in range(len(predicted)):
            if predicted[i] == test_Y[i]:
                correct += 1
        print(f"Accuracy: {correct / float(len(predicted)) * 100} %")
