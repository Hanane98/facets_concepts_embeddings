import argparse
import pickle
from sklearn.cluster import KMeans, AffinityPropagation
import hdbscan
import torch
import numpy as np

def load_embeddings(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def perform_clustering(embeddings, num_clusters, method):
    embeddings = np.array(embeddings)  # Convert embeddings to a NumPy array if not already
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
        return kmeans.cluster_centers_
    elif method == 'affinity':
        from sklearn.cluster import AffinityPropagation
        affinity = AffinityPropagation(random_state=0).fit(embeddings)
        return affinity.cluster_centers_
    elif method == 'hdbscan':
        import hdbscan
        hdb = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, cluster_selection_epsilon=0.5).fit(embeddings)
        unique_clusters = set(hdb.labels_)
        centroids = []
        for cluster in unique_clusters:
            if cluster != -1:  # Ignore noise points
                cluster_points = embeddings[hdb.labels_ == cluster]
                centroid = cluster_points.mean(axis=0)
                centroids.append(centroid)
        return centroids


'''
def perform_clustering(embeddings, num_clusters, method):
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
        return kmeans.cluster_centers_
    elif method == 'affinity':
        affinity = AffinityPropagation(random_state=0).fit(embeddings)
        return affinity.cluster_centers_
    elif method == 'hdbscan':
        # Adjust these parameters to control the clustering outcome
        hdb = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, cluster_selection_epsilon=0.5).fit(embeddings)
        # Generate centroids for HDBSCAN clusters
        unique_clusters = set(hdb.labels_)
        centroids = []
        for cluster in unique_clusters:
            print('.')
            if cluster != -1:  # Ignore noise points
                cluster_points = embeddings[hdb.labels_ == cluster]
                centroid = cluster_points.mean(axis=0)
                centroids.append(centroid)
        return centroids
'''

def save_centroids(centroids, file_path):
    centroids_tensor = torch.tensor(centroids)
    torch.save(centroids_tensor, file_path)

def main(input_file, output_file, num_clusters, method):
    # Load embeddings
    embeddings_dict = load_embeddings(input_file)
    embeddings = list(embeddings_dict.values())

    # Cluster embeddings
    centroids = perform_clustering(embeddings, num_clusters, method)

    # Save centroids
    save_centroids(centroids, output_file)
    print(f"Centroids saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster embeddings and save centroids.")
    parser.add_argument('--input', type=str, required=True, help='Path to input .pkl file with embeddings')
    parser.add_argument('--output', type=str, required=True, help='Path to output .pt file to save centroids')
    parser.add_argument('--clusters', type=int, default=10, help='Number of clusters for KMeans (ignored for Affinity Propagation and HDBSCAN)')
    parser.add_argument('--method', type=str, choices=['kmeans', 'affinity', 'hdbscan'], default='kmeans', help='Clustering method: kmeans, affinity, or hdbscan')

    args = parser.parse_args()

    main(args.input, args.output, args.clusters, args.method)

