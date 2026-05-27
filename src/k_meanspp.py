import numpy as np
from scipy.spatial import distance

"""
K-means++ clustering algorithm.

Based off of the code in:

- K-means++ Algorithm - ML
https://www.geeksforgeeks.org/machine-learning/ml-k-means-algorithm/

Eynar Ason Eklöf
eaeklof@kth.se

2026-05-27
"""

# INPUT: 
# 
# in_data: Array of N-d datapoints, i.e. a list of numpy arrays. Should be normalized!
#
# k: Number of partitions *k*.
#
# rng: Random number generator object.
#
# OUTPUT:
# 
# cluster_assignments: Membership array modeling the partitioning of input data
# into clusters.
# centroids: Centerpoint (mean vector) of each cluster.
# nr_iterations: Number of iterations before convergence

def k_meanspp(k, in_data, rng):
    dimensions = in_data[0].size
    converged = False
    centroids = np.zeros((k, dimensions))
    n_iterations = 0
    cluster_assignments = np.zeros(in_data.shape[0], dtype='int64')
    converge_threshold = 1e-3

    centroids[0] = in_data[rng.integers(in_data.shape[0])]
    
    #distances = []
    #for datapoint in in_data:
    #    distances.append(distance.sqeuclidean(centroids[0], datapoint))
    distances = distance.cdist(in_data, [centroids[0]], metric='sqeuclidean')

    probabilities = np.ndarray.flatten(distances) / np.sum(distances)
    choices = rng.choice(in_data, size=k-1, p=probabilities)
    for i in range(1, k):
        centroids[i] = choices[i-1]
    
    # Free memory (unsure if this is actually necessary)
    distances = None

    while not converged:
        n_iterations += 1
        converged_centroids = 0
        
        # Step 1: Assign points to nearest centroids
        for i in range(in_data.shape[0]): # for each datapoint in the input data
                
            # Initialize the calculation by doing the first iteration outside the loop
            distance_to_centroid = distance.sqeuclidean(in_data[i], centroids[0])
            
            for j in range(1, k):
                new_distance_to_centroid = distance.sqeuclidean(in_data[i], centroids[j])
                if (new_distance_to_centroid < distance_to_centroid):
                    distance_to_centroid = new_distance_to_centroid
                    cluster_assignments[i] = j

        # Step 2: recompute the centroids
        vector_sums = np.zeros((k, dimensions))
        assignment_counts = np.zeros(k)
        for (assignment, data_point) in zip(cluster_assignments, in_data):
            vector_sums[assignment] += data_point
            assignment_counts[assignment] += 1

        new_centroids = ( 
                np.where(vector_sums == np.zeros(dimensions), 
                         in_data[rng.integers(in_data.shape[0])], 
                         vector_sums) 
                /   
                np.where(assignment_counts[:, None] == 0,
                         [1],
                         assignment_counts[:, None])
        )   

        deltas = np.array(list(map(distance.euclidean, centroids, new_centroids)))
    
        if np.sum(deltas < converge_threshold) == k:
            converged = True

        centroids = new_centroids
       
    return (cluster_assignments, centroids, n_iterations)
