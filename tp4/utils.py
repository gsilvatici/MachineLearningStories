import numpy as np
import math
import sys
import matplotlib.pyplot as plt

def euclidean (v1) :
    def f (v2) :
        # return np.linalg.norm(v1-v2)
        result = np.sqrt(np.sum(np.square(v1 - v2)))
        assert math.isnan(result) is False
        return result

    return f


def pairwise_euclidean (elements) :
    distances = []
    if len(elements) == 1 :
        distances.append(0)
    else :
        for i in range(len(elements)) :
            distance = euclidean(elements[i])
            for j in np.arange(i + 1, len(elements)) :
                distances.append(distance(elements[j]))
    return np.array(distances)



def get_sample_cluster (clusters, centroids) :
    def f (sample) :
        distance = euclidean(sample)
        min_distance = -1
        closest_cluster = -1
        for cluster, centroid in zip(clusters, centroids) :
            d = distance(centroid)
            if min_distance == -1 or d < min_distance :
                min_distance = d
                closest_cluster = cluster
        return closest_cluster

    return f