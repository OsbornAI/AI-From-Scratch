import numpy as np
import random

class KMeansClustering:
    def __init__(self, k):
        self.k = k
        self.centroids = None

    def __eDistance(self, arr1, arr2):
        difference = arr1 - arr2
        distance = np.sqrt(np.dot(difference, difference))

        return distance

    def fit(self, train_x, iterations):
        clusters_centroids = []

        for _ in range(iterations):
            centroids = np.array(random.sample(list(train_x), self.k)) # This will represent our centroid points

            while True:
                clusters = [[] for _ in range(self.k)] # This will store all of our clusters

                for point in train_x:
                    distances = []
                    for i, centroid in enumerate(centroids):
                        distance = self.__eDistance(point, centroid)
                        distances.append((i, distance))

                    distances.sort(key=lambda x: x[1])
                    
                    cluster = distances[0][0]
                    clusters[cluster].append(point)

                new_centroids = np.array([np.array(cluster).mean(axis=0) for cluster in clusters])

                # print(f"\nCentroids: {centroids} | New Centroids: {new_centroids}")

                if np.all(centroids == new_centroids):
                    centroids = new_centroids
                    clusters_centroids.append((clusters, centroids))
                    break

                centroids = new_centroids

        # Now I want to calculate the sum of the variances for all of the clusters and then the one with the lowest summed variance will be our good model

        variances = [sum([np.var(item) for item in sublist[0]]) for sublist in clusters_centroids]
        key = np.argmin(variances)
        best_centroids = clusters_centroids[key][1]

        self.centroids = best_centroids

    def cluster(self, point):
        distances = [self.__eDistance(point, centroid) for centroid in self.centroids]
        label = np.argmin(distances)

        return label

if __name__ == '__main__':

    train_x = np.random.random(size=(10, 2))

    model = KMeansClustering(3)

    model.fit(train_x, 10)

    # Now I want to go through and test to see if my model works or not
    print(model.cluster(train_x[2]))