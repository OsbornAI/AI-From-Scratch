import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class KMeansClustering:
    def __init__(self, k):
        self.k = k
        self.centroids = None

    def __eDistance(self, arr1, arr2):
        difference = arr1 - arr2
        distance = np.sqrt(np.dot(difference, difference))

        return distance

    def fit(self, points, iterations):
        clusters_centroids = []

        for _ in range(iterations):
            centroids = np.array(random.sample(list(points), self.k)) # This will represent our centroid points

            while True:
                clusters = [[] for _ in range(self.k)] # This will store all of our clusters

                for point in points:
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
    points = np.random.random(size=(1000, 2))

    model = KMeansClustering(5) # Hangon, why am I getting less classified values?
    model.fit(points, 10)

    points_x = [x for x, y in points]
    points_y = [y for x, y in points]
    df = pd.DataFrame()
    df['x'] = points_x
    df['y'] = points_y

    labels = [model.cluster(df.iloc[i, :]) for i in range(len(df.index))]
    df['label'] = labels

    unique_labels = df['label'].unique()
    for label in unique_labels:
        target_df = df[df['label'] == label]
        plt.scatter(target_df['x'], target_df['y'])

    plt.savefig('fig1.png')