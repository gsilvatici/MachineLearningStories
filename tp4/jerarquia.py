import numpy as np
import sys
from utils import get_sample_cluster

class HierarchicalClustering:

    #k es el numero de cluster donde parar
    def __init__(self, samples, k):
        self.samples = samples
        self.k = k
        self.cluster_per_sample = np.arange(0, len(self.samples))

        self.cluster_classifications = None
        self.clusters = None
        self.centroids = None
        self.distances = np.zeros((len(samples), len(samples)))
        # inicializar matriz de distancias


    def get_centroids(self, clusters):
        centroids = []
        for c in clusters:
            cluster_elems_indexes = np.where(self.cluster_per_sample == c)[0]
            centroid = self.samples[cluster_elems_indexes].mean(axis=0)
            centroids.append(centroid)
        return centroids

    def predict(self, samples):
        predictor = get_sample_cluster(self.clusters, self.centroids)
        winner_clusters = list(map(predictor, samples))
        predictions = list(map(lambda c: self.cluster_classifications[self.clusters.index(c)], winner_clusters))
        return predictions
