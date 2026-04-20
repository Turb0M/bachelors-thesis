import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from k_means_standard import k_means

"""
Main module for Eynar Eklöf's and Hannes Hultin's Bachelor's thesis project.

Compares the performance of different K-means algorithm implementations.

Eynar Ason Eklöf
eaeklof@kth.se

2026-04-20
"""

def main():
    print("Initializing tools/variables")
    rng_seed = 1234
    scaler = StandardScaler()
    pca = PCA(n_components = 10)
    
    print("Loading dataset")
    raw_data = pd.read_csv('kidney_sc_dataset.csv')
    
    print("Pre-processing dataset")
    data_trimmed = raw_data.select_dtypes(include=['int64'])
    data_scaled = pd.DataFrame(scaler.fit_transform(data_trimmed))
    pca.fit(data_scaled)
    data_reduced = pd.DataFrame(pca.transform(data_scaled))
    data_normalized = data_reduced.copy()
    
    for column in data_normalized:
        data_normalized[column] = data_normalized[column] / data_normalized[column].abs().max()

    print(data_normalized.head())

    input_data = data_normalized.to_numpy()

    print("Running k-means")
    partitions, centroids = k_means(5, input_data, rng_seed)

    print(centroids)

if __name__ == '__main__':
    main()
