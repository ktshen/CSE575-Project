"""
In this part, you are required to implement the k-means algorithm and apply your implementation on the given dataset, which contains a set of 2-D points. You are required to implement two different strategies for choosing the initial cluster centers.

Strategy 1: randomly pick the initial centers from the given samples.

Strategy 2: pick the first center randomly; for the i-th center (i>1), choose a sample (among all possible samples) such that the average distance of this chosen one to all previous (i-1) centers is maximal.

You need to test your implementation on the given data, with the number k of clusters ranging from 2-10. Plot the objective function value vs. the number of clusters k. Under each strategy, plot the objective function twice, each start from a different initialization.

(Referring to the course notes:  When clustering the samples into k clusters/sets Di, with respective center/mean vectors 𝜇1, 𝜇2, … 𝜇k,  the objective function is defined as
ΣΣ || x - 𝜇 ||^2
"""

from scipy.io import loadmat
import numpy as np
import argparse
import matplotlib.pyplot as plt
import random


class KMeanClassifier:
    def __init__(self, X, iter=1000, strategy=1):
        self.X = X
        self.iter = iter
        self.strategy = strategy

    def run_and_plot(self, K, show_or_store=1):
        obj_values_for_each_K_with_two_times = np.zeros((K+1)*2).reshape(2, K+1)
        for i in range(2):
            obj_values_for_each_K_with_two_times[i] = self.run(K)
        self.plot_obj_value(obj_values_for_each_K_with_two_times, show_or_store=show_or_store)

    def run(self, K):
        obj_value_for_each_K = np.zeros(K+1)
        for k in range(2, K+1):
            self.K = k
            obj_value = self.kmeans()
            obj_value_for_each_K[k] = obj_value
        return obj_value_for_each_K

    def kmeans(self):
        previous_centroids, centroids, labels = None, None, None
        # Initialize centroids
        if self.strategy == 1:
            centroids = self.random_initialize_centroids()
        elif self.strategy == 2:
            centroids = self.initialize_centroids_with_max_distance_between()
        else:
            raise ValueError("Wrong strategy number")

        # classify sample to nearest centroid and recompute the centroid according to the classification
        for iter in range(self.iter):
            labels = self.classify_to_nearest_centroids(centroids)
            previous_centroids = centroids
            centroids = self.recompute_centroids(previous_centroids, labels)
            if np.array_equal(previous_centroids, centroids):
                break

        obj_value = self.compute_obj_function_value(centroids, labels)
        return obj_value

    def compute_obj_function_value(self, centroids, labels):
        value = 0

        for i in range(self.K):
            samples_in_cluster_i = self.X[(labels==i)]
            u = centroids[i]
            squared = np.square(u - samples_in_cluster_i)
            value += np.sum(squared)
        return value

    def random_initialize_centroids(self):
        random_index = np.random.permutation(self.X.shape[0])[:self.K]
        return self.X[random_index]

    def initialize_centroids_with_max_distance_between(self):
        """
            Pick the first centroid and based on that centroid find the most distant sample from it and assign it as a centroid too.
            If that sample is already assigned as centroid, then pick the second distant sample and so on.
        """
        centroid_index = np.array([-1 for i in range(self.K)])
        centroid_index[0] = random.randint(0, self.X.shape[0])

        for k in range(2, self.K+1):
            init_matrix = np.repeat(self.X, k-1, axis=1).reshape(self.X.shape[0], k-1, 2)
            square = np.square(init_matrix - self.X[centroid_index[:k-1].astype(int)])
            # print(init_matrix.shape, self.X[centroid_index[:k-1].astype(int)].shape)
            sum = np.sum(square, axis=2)
            sqrt = np.sqrt(sum)
            # mean_distance_to_each_centroid = np.true_divide(np.sum(sqrt, axis=1), k-1)
            mean_distance_to_each_centroid = np.mean(sqrt, axis=1)
            sorted_indices = np.argsort(-mean_distance_to_each_centroid)
            for idx in sorted_indices:
                if not idx in centroid_index:
                    centroid_index[k-1] = idx
                    break
        return self.X[centroid_index.astype(int)]

    def classify_to_nearest_centroids(self, centroids):
        m_data = self.X.shape[0]
        distance_matrix = np.zeros(shape=(m_data, self.K))

        for i in range(self.K):
            u = centroids[i]
            squared = np.square(self.X - u)
            distance = np.sqrt(np.sum(squared, axis=1))
            distance_matrix[:, i] = distance
        return np.argmin(distance_matrix, axis=1)

    def recompute_centroids(self, centroids, labels):
        new_centroids = np.zeros(shape=centroids.shape)
        for i in range(self.K):
            samples_in_cluster_i = self.X[(labels==i)]
            new_centroids[i] = np.mean(samples_in_cluster_i, axis=0)
        return new_centroids

    def plot_obj_value(self, obj_values, show_or_store=1):
        x = np.arange(obj_values[0].shape[0])
        figure = plt.figure()
        plt.plot(x, obj_values[0], 'b')
        plt.plot(x, obj_values[1], 'r')
        plt.title("Value of Objective Function")
        plt.xlabel("K")
        plt.ylabel("Value")
        plt.xticks(np.arange(1, 11, 1))
        if show_or_store == 1:
            plt.show()
        else:
            figure.savefig(f"strategy-{self.strategy}.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--strategy",
        type=int,
        help="strategy to be used, 1 or 2",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="file to read",
        default="AllSamples.mat"
    )
    parser.add_argument(
        "-i",
        "--img",
        type=int,
        help="Show image or store image. 1 for showing, 2 for storing",
        default=1,
    )

    args = parser.parse_args()
    strategy = args.strategy
    DATASET_FILE = args.file
    img_showing_or_storing = args.img

    dataset = (loadmat(DATASET_FILE))["AllSamples"]
    classifier = KMeanClassifier(dataset, iter=3000, strategy=strategy)
    classifier.run_and_plot(K=10, show_or_store=img_showing_or_storing)
