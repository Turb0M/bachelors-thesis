import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from k_means_standard import k_means
from k_meanspp import k_meanspp
from e_k_means import e_k_means

"""
Main module for Eynar Eklöf's and Hannes Hultin's Bachelor's thesis project.

Compares the performance of different K-means algorithm implementations.

Eynar Ason Eklöf
eaeklof@kth.se

2026-05-11
"""

matplotlib.use("Agg")

def main(variant, k):
    if k:
        print(f"\nRunning experiment on variant \"{variant}\" and k={k}\n")
    else:
        print(f"\nRunning experiment on variant \"{variant}\"\n")

    rng = np.random.default_rng(seed=1234)
    scaler = StandardScaler()
    pca = PCA(n_components = 10)
    
    print("Loading dataset")
    raw_data = pd.read_csv('kidney_sc_dataset.csv')
    
    print("Pre-processing dataset")
    data_trimmed = raw_data.select_dtypes(include=['int64'])
    data_scaled = pd.DataFrame(scaler.fit_transform(data_trimmed))
    pca.fit(data_scaled)
    data_preprocessed = pd.DataFrame(pca.transform(data_scaled))
    
    # print(data_preprocessed.head()) # debug

    input_data = data_preprocessed.to_numpy()

    print("Running k-means")

    if variant == "standard":
        partitions, centroids = k_means(k, input_data, rng)
    elif variant == "kpp":
        partitions, centroids = k_meanspp(k, input_data, rng)
    elif variant == "enhanced":
        partitions, centroids = e_k_means(k, input_data, rng)
    else:
        print("No variant to test provided")
        return 1

    # Scoring
    global_centroid = np.sum(input_data, axis=0) / input_data.shape[0]
    
    ## Calinski-Harabasz Index
    bcss = 0.0
    wcss = 0.0
    for i in range(len(partitions)):
        bcss += (
            partitions[i].shape[0] 
            * distance.sqeuclidean(centroids[i], global_centroid)
        )
    for i in range(len(partitions)):
        for j in range(partitions[i].shape[0]):
            wcss += (
                distance.sqeuclidean(partitions[i][j], centroids[i])
            )
    chi = (bcss / (k - 1)) / (wcss / (input_data.shape[0] - k))
    
    ## TODO: Davies-Bouldin Index
    
    print("\nCalinski-Harabasz Score: ", chi, "\n")

    # Plotting
    colors = ['red', 'orange', 'yellow', 
              'green', 'blue', 'purple', 
              'pink', 'brown', 'cyan']

    # Plot 1% of the first two PCA components of each partition
    partitions_sample = [
        rng.choice(
            partition[:,0:2], 
            size=partition.shape[0]//100, 
            replace=False
        ) 
        for partition in partitions
    ]
    
    for i in range(len(partitions_sample)):
        plt.scatter(
            partitions_sample[i][:,0], 
            partitions_sample[i][:,1], 
            color=colors[i],
            marker=".",
            s=2.0
        )
    
    for i in range(centroids.shape[0]):
        plt.scatter(centroids[i][0], centroids[i][1], color='black', marker='X')
    
    plt.scatter(global_centroid[0], global_centroid[1], color='black', marker='*')
    
    # plt.show()
    if k:
        plt.savefig(f"kmeans-{variant}-k-{k}-clustering.png", dpi=300)
    else:
        plt.savefig(f"kmeans-{variant}-clustering.png", dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test K-means variants")
    parser.add_argument("-v", "--variant", help="The K-means variant to be tested")
    parser.add_argument("-k", help="Number of clusters (if required for variant)")
    args = parser.parse_args()
    main(args.variant, int(args.k))
