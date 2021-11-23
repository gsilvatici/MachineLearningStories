import numpy as np
import matplotlib.pyplot as plt
from utils import pairwise_euclidean, get_sample_cluster

class Kmeans:

    def __init__(self, k, samples):
        self.k = k
        self.samples = samples
        self.cluster_per_sample = np.random.randint(0, k, len(samples))

        # se inicializa al llamar add_cluster_classification
        self.cluster_classifications = None
        self.clusters = None
        self.centroids = None

    def get_centroids(self, clusters):
        centroids = []
        for c in clusters:
            cluster_elems_indexes = np.where(self.cluster_per_sample == c)[0]
            centroid = self.samples[cluster_elems_indexes].mean(axis=0)
            centroids.append(centroid)
        return centroids

    def run(self, epochs):
        for j in range(epochs):
            clusters = np.unique(self.cluster_per_sample)
            
            # arreglo de centroides para cada agrupacion
            centroids = self.get_centroids(clusters)

            # self.compute_variances(clusters)

            # calcular agrupacion para cada observacion
            current_clusters_per_sample = np.fromiter(
                map(get_sample_cluster(clusters, centroids), self.samples), dtype=int
            )
            if (self.cluster_per_sample == current_clusters_per_sample).all():
                return
            self.cluster_per_sample = current_clusters_per_sample
            print(f'Finished epoch {j}')

    # this should be called only when finishing run
    def add_cluster_classification(self, labels):
        cluster_classifications = []
        clusters = np.unique(self.cluster_per_sample)
        for c in clusters:
            cluster_elems_indexes = np.where(self.cluster_per_sample == c)[0]
            cluster_labels = labels[cluster_elems_indexes]
            print(f'Counts for cluster {c} are {np.bincount(cluster_labels[:, 0])}')
            winner = np.bincount(cluster_labels[:, 0]).argmax()
            cluster_classifications.append(winner)

        self.cluster_classifications = cluster_classifications
        self.clusters = clusters.tolist()
        self.centroids = self.get_centroids(clusters)

    def predict(self, samples):
        predictor = get_sample_cluster(self.clusters, self.centroids)
        winner_clusters = list(map(predictor, samples))
        predictions = list(map(lambda c: self.cluster_classifications[self.clusters.index(c)], winner_clusters))
        return predictions

