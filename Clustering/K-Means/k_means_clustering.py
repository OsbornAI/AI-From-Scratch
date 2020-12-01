import numpy as np # Maths
import pandas as pd
import matplotlib.pyplot as plt
import random

# This will be our KMeansClustering model
class KMeansClustering:
    def __init__(self, k):
        self.k = k # Save the value of k (number of clusters)
        self.centroids = None # This will later store our optimized centroids or cluster positions

    # This will train our model on the data and will find the optimal positions for the centroids
    def fit(self, points, iterations):
        clusters_centroids = [] # This will contain the clustered data and centroids for each iteration to later be used to find the centroids that result in the lowest variance

        # This will determine how many centroid-cluster pairs to compare for finding the lowest variance
        for _ in range(iterations):
            centroids = np.array(random.sample(list(points), self.k)) # This will represent our centroid points, initialized randomly from points in our data

            # Loop until our clusters do not change
            while True:
                clusters = [[] for _ in range(self.k)] # This will store the clusters thaqt belong to each centroid

                # Iterate over all of the points in the data and find their closest cluster, appending it to the cluster of which it is closest to
                for point in points:
                    distances = []
                    for i, centroid in enumerate(centroids):
                        distance = np.linalg.norm(point - centroid)
                        distances.append((i, distance))

                    # Find the cluster to which the point has the shortest distance
                    distances.sort(key=lambda x: x[1])
                    
                    # Group this point into its respective cluster
                    cluster = distances[0][0]
                    clusters[cluster].append(point)

                # This will find the mean of all of the points in our set resulting in finding our new centroids
                new_centroids = np.array([np.array(cluster).mean(axis=0) for cluster in clusters])

                # If our centroids/clusters have not changed, then we will append a cluster-centroid pair to compare the variances later on and break this while loop
                if np.all(centroids == new_centroids):
                    centroids = new_centroids
                    clusters_centroids.append((clusters, centroids))
                    break

                # If our clusters have changed then we will set our main centroids to be our newly calculated centroids
                centroids = new_centroids


        # Find the centroid that results in the lowest total variance from the clusters
        variances = [sum([np.var(item) for item in sublist[0]]) for sublist in clusters_centroids]
        key = np.argmin(variances)
        best_centroids = clusters_centroids[key][1]

        # Set our models centroids to be the centroids that performed the best during training
        self.centroids = best_centroids

    # This function will classify a point to it's nearest cluster
    def cluster(self, point):
        # Calculate the distances of the point to all of the clusters and return the cluster that has the shortest distance
        distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
        label = np.argmin(distances)

        return label

if __name__ == '__main__':
    points = np.random.random(size=(1000, 2)) # Generate 1000 2d points ranging between 0,0 to 1,1

    k = 5 # Determine how many clusters we want to cluster the data into
    model = KMeansClustering(k) # Create our model
    model.fit(points, 10) # Fit the model to the data

    # Put our randomly generated points into a dataframe
    points_x = [x for x, y in points]
    points_y = [y for x, y in points]
    df = pd.DataFrame()
    df['x'] = points_x
    df['y'] = points_y

    # Label all of the points in the dataframe
    labels = [model.cluster(df.iloc[i, :]) for i in range(len(df.index))]
    df['label'] = labels

    # Plot all of the points, colouring them based on what cluster they belong to
    unique_labels = df['label'].unique()
    for label in unique_labels:
        target_df = df[df['label'] == label]
        plt.scatter(target_df['x'], target_df['y'])

    # Save the figure to a png to visually observe the clustering
    plt.savefig('fig1.png')