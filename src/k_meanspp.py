import numpy as np
from scipy.spatial import distance

"""
K-means++ clustering algorithm.

Based off of the code in:

- K-means++ Algorithm - ML
https://www.geeksforgeeks.org/machine-learning/ml-k-means-algorithm/

Eynar Ason Eklöf
eaeklof@kth.se

2026-05-06
"""

# INPUT: 
# 
# in_data: Array of N-d datapoints, i.e. a list of numpy arrays. Should be normalized!
#
# k: Number of partitions *k*.
#
# rng_seed: self explanitory.
#
# OUTPUT:
# 
# clusters: Partition of the datapoints in `in_data`. i.e. a 2-d list of numpy arrays.
# centroids: Centerpoint of each partition in `clusters`.

def k_meanspp(k, in_data, rng):
    dimensions = in_data[0].size
    converged = False
    centroids = np.zeros((k, dimensions))
    n_iterations = 0

    centroids[0] = in_data[rng.integers(in_data.shape[0])]
    distances = []
    for datapoint in in_data:
        distances.append(distance.sqeuclidean(centroids[0], datapoint))
    probabilities = np.array(distances) / np.sum(distances)
    choices = rng.choice(in_data, size=k-1, p=probabilities)
    for i in range(1, k):
        centroids[i] = choices[i-1]

    #print(centroids) # debug

    while not converged:
        n_iterations += 1
        converged_centroids = 0
        partitions = [ [] for _ in range(k) ]
        
        # Step 1: Assign points to nearest centroids
        for i in range(in_data.shape[0]): # for each datapoint in the input data
                
            # Initialize the calculation by doing the first iteration outside the loop
            distance_to_centroid = distance.sqeuclidean(in_data[i], centroids[0])
            assigned_cluster_index = 0
            
            for j in range(1, k):
                new_distance_to_centroid = distance.sqeuclidean(in_data[i], centroids[j])
                if (new_distance_to_centroid < distance_to_centroid):
                    distance_to_centroid = new_distance_to_centroid
                    assigned_cluster_index = j
                
            partitions[assigned_cluster_index].append(in_data[i])

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

    partitions = [ np.array(partition) for partition in partitions ]
    return (partitions, centroids, n_iterations)
