import numpy as np
from scipy.spatial import distance

"""
Unsupervised K-means (U-k-means) clustering algorithm.

Based off of the pseudocode in:

- Unsupervised K-Means Clustering Algorithm 
(DOI: 10.1109/ACCESS.2020.2988796)

Eynar Ason Eklöf
eaeklof@kth.se

2026-05-06
"""

# INPUT: 
# 
# in_data: Array of N-d datapoints, i.e. an array of numpy arrays. 
# rng: Random number generator object
#
# OUTPUT:
# 
# clusters: Partition of the datapoints in `in_data`. i.e. a 2-d list of numpy arrays.
# centroids: Centerpoint of each partition in `clusters`.

def u_k_means(in_data, rng):
    # Step 1: Initialization
    epsilon = 1e-3
    dimensions = in_data[0].size
    centroids = np.copy(in_data)
    converged = False
    proportions = [ 1/in_data.shape[0] for _ in range(in_data.shape[0]) ]
    new_proportions = proportions.copy()
    beta, gamma = 1, 1
    iteration_nr = 0
    # print(centroids) # debug

    while not converged:
        print("Not converged.") # testing
        converged_centroids = 0
        partitions = [ [] for _ in range(in_data.shape[0]) ]
        
        # Step 2: Assign points to nearest centroids
        for i in range(in_data.shape[0]): # for each datapoint in the input data
            # Initialize the calculation by doing the first iteration outside the loop
            distance_to_centroid = (
                distance.sqeuclidean(in_data[i], centroids[0])
                - gamma * np.log(proportions[0])
            )
            assigned_cluster_index = 0
            
            for j in range(1, len(centroids)):
                new_distance_to_centroid = (
                    distance.sqeuclidean(in_data[i], centroids[j])
                    - gamma * np.log(proportions[j])
                )
                if (new_distance_to_centroid < distance_to_centroid):
                    distance_to_centroid = new_distance_to_centroid
                    assigned_cluster_index = j
                
            partitions[assigned_cluster_index].append(in_data[i])

        # Step 3: recompute gamma
        gamma = exp(len(centroids) / 250)

        # Step 4: update proportions
        for i in range(len(proportions)):
            for j in range(in_data.shape[0]):
                new_proportions[i] += int((in_data[j] in partitions[i]))
            new_proportions[i] += (
                (beta / gamma) * proportions[i]
                * (np.log(proportions[i]) 
                - sum([ proportion * np.log(proportion) for proportion in proportions ]))
            )

        # Step 5: update beta
        eta = min(1, 1/np.pow(iteration_nr, np.floor(dimensions / 2 - 1)))
        beta_criteria_1 = (
            sum([ np.exp(
                -eta 
                * in_data.shape[0] 
                * np.abs(new_proportion - proportion)
            ) for new_proportion, proportion in zip(new_proportions, proportions) ]) 
            / len(partitons)
        )
        beta = min()

        # Step 2: recompute the centroids
        for i in range(k):
            vector_sum = np.zeros(dimensions)
            for j in range(len(partitions[i])):
                vector_sum += partitions[i][j]
            
            new_centroid = vector_sum / np.int64(len(partitions[i]))
            # testing
            print(distance.euclidean(centroids[i], new_centroid)) 
            if distance.euclidean(centroids[i], new_centroid) < epsilon:
                converged_centroids += 1

            centroids[i] = new_centroid

        if converged_centroids == k:
            converged = True
            print("Converged!")

        iteration_nr += 1

    partitions = [ np.array(partition) for partition in partitions ]
    return (partitions, centroids)
