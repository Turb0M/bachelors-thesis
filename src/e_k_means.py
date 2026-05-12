import numpy as np
from scipy.spatial import distance

"""
"Enhanced K-means" clustering algorithm (Fahim's Method).

Based off of the algorithm presented in:

- An efficient enhanced k-means clustering algorithm
(DOI: https://doi.org/10.1631/jzus.2006.A1626)

Eynar Ason Eklöf
eaeklof@kth.se

2026-05-12
"""

# INPUT: 
# 
# in_data: Array of N-d datapoints, i.e. a list of numpy arrays. Should be normalized!
# Update: Using a Pandas DataFrame for this instead
#
# k: Number of partitions *k*.
#
# rng_seed: self explanitory.
#
# OUTPUT:
# 
# clusters: Partition of the datapoints in `in_data`. i.e. a 2-d list of numpy arrays.
# centroids: Centerpoint of each partition in `clusters`.

def e_k_means(k, in_data, rng):
    dimensions = in_data[0].size
    centroids = rng.random((k, dimensions)) - 0.5
    converged = False
    point_distances = np.full(in_data.shape[0], np.finfo(np.float64).max)
    cluster_ids = np.zeros(in_data.shape[0], dtype=int)
    n_iterations = 0

    while not converged:
        converged_centroids = 0
        partitions = [ [] for _ in range(k) ]
        
        # Step 1: Assign points to nearest centroids
        for i in range(in_data.shape[0]): # for each datapoint in the input data
            # Enhanced K-means: skip iterations
            if (
                n_iterations > 1
                and point_distances[i] 
                >= distance.sqeuclidean(in_data[i], centroids[cluster_ids[i]])
            ):
                point_distances[i] = distance.sqeuclidean(in_data[i], centroids[cluster_ids[i]])
                partitions[cluster_ids[i]].append(in_data[i])
                continue
            
            # reset the distance to nearest centroid in case distance increases
            point_distances[i] = np.finfo(np.float64).max
            
            for j in range(k):
                dist = distance.sqeuclidean(in_data[i], centroids[j]) 
                if (dist < point_distances[i]):
                    point_distances[i] = dist
                    cluster_ids[i] = j
                
            partitions[cluster_ids[i]].append(in_data[i])

        # Step 2: recompute the centroids
        for i in range(k):
            vector_sum = np.zeros(dimensions)
            for j in range(len(partitions[i])):
                vector_sum += partitions[i][j]
            
            new_centroid = vector_sum / np.int64(len(partitions[i]))
            if distance.euclidean(centroids[i], new_centroid) < 1e-3:
                converged_centroids += 1

            centroids[i] = new_centroid
        
        if converged_centroids == k:
            converged = True

        n_iterations += 1
    
    partitions = [ np.array(partition) for partition in partitions ]
    return (partitions, centroids, n_iterations)
