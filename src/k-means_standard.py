import numpy as np
from scipy.spatial import distance

"""
Standard K-means clustering algorithm.

Based off of the pseudocode in:

- K-means clustering algorithms: A comprehensive review, variants analysis, and 
advances in the era of big data 
(DOI: https://doi.org/https://doi.org/10.1016/j.ins.2022.11.139)

- K-means clustering
(https://en.wikipedia.org/w/index.php?title=K-means_clustering&oldid=1348629839)

Eynar Ason Eklöf
eaeklof@kth.se

2026-04-14
""""

# INPUT: 
# in_data: Array of N-d datapoints, i.e. a list of lists. Should be normalized!
# k: Number of partitions *k*.
#
# OUTPUT:
# clusters: Partition of the datapoints in `in_data`. i.e. a list of 2-d lists.
# centroids: Centerpoint of each partition in `clusters`.

my_seed = 1234

def k-means(k, in_data):
    rng = np.random.default_rng(seed=my_seed)
    dimensions = in_data[0].size
    centroids = rng.random(k, dimensions)
    partition_assignments = np.zeros(in_data.size, dtype=np.int32)

    for i in range(in_data.size):
        distance = distance.cdist(in_data[i], centroids[0], 'sqeuclidean')
        assigned_cluster_index = 0
        
        for j in range(1, k):
            new_distance = distance.cdist(in_data[i], centroids[j], 'sqeuclidean')
            if (new_distance < distance):
                distance = new_distance
                assigned_cluster_index = j
        
        partition_assignments[i] = j        
