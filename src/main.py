import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from k_means_standard import k_means

"""
Main module for Eynar Eklöf's and Hannes Hultin's Bachelor's thesis project.

Compares the performance of different K-means algorithm implementations.

Eynar Ason Eklöf
eaeklof@kth.se

2026-05-05
"""

def main():
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
    
    print(data_preprocessed.head())

    input_data = data_preprocessed.to_numpy()

    print("Running k-means")
    partitions, centroids = k_means(2, input_data, rng)

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
    
    plt.show()

if __name__ == '__main__':
    main()
