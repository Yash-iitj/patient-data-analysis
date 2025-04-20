import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def label_dropoffs_by_kmeans():
    """
    Reads './data/final_data_small.csv', adds an 'IS_DROPOFF' column using KMeans clustering,
    and overwrites the same file. The smaller cluster (between 25% and 30%) is labeled as 1 (drop-off).
    """
    file_path = "./data/final_data_small.csv"
    df = pd.read_csv(file_path)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # KMeans loop
    kmeans = KMeans(n_clusters=2, random_state=42)
    while True:
        clusters = kmeans.fit_predict(X_scaled)
        df['cluster'] = clusters
        cluster_sizes_percentage = df['cluster'].value_counts(normalize=True) * 100
        cluster_sizes_percentage.sort_index(inplace=True)
        if 25 < cluster_sizes_percentage.iloc[1] < 30:
            break

    # Assign drop-off label
    dropoff_cluster = cluster_sizes_percentage.idxmin()
    df['IS_DROPOFF'] = df['cluster'].apply(lambda x: 1 if x == dropoff_cluster else 0)
    df.drop(columns=['cluster'], inplace=True)

    # Overwrite the file
    df.to_csv(file_path, index=False)
