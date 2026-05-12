import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

"""
preprocessing.py

Module for the k-means pre-processing pipeline used in Eynar Eklöf's and 
Hannes Hultin's bachelor's thesis project.

eaeklof@kth.se
2026-05-12
"""

def prepare(data_file: str):
    scaler = StandardScaler()
    pca = PCA(n_components = 10)
   
    print(f"Loading dataset {data_file}")
    raw_data = pd.read_csv(data_file)
    
    print("Pre-processing dataset")
    data_trimmed = raw_data.select_dtypes(include=['int64'])
    data_scaled = pd.DataFrame(scaler.fit_transform(data_trimmed))
    pca.fit(data_scaled)
    data_dim_reduced = pd.DataFrame(pca.transform(data_scaled))
    
    data_preprocessed = data_dim_reduced.to_numpy()

    return data_preprocessed
