import argparse
import pickle
import torch
import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import silhouette_score
import hdbscan

def load_centroids(file_path):
    return torch.load(file_path)

def load_embeddings(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def calculate_vectors(centroid, embeddings):
    vectors = {}
    for key, embedding in embeddings.items():
        dot_product = np.multiply(embedding, centroid)
        norm_product = np.linalg.norm(dot_product)
        vectors[key] = dot_product / norm_product if norm_product != 0 else 0
    return vectors


def find_optimal_clusters(embeddings):
    silhouette_scores = []
    for n_clusters in range(10, 200):  # Range from 10 to 200 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
        score = silhouette_score(embeddings, kmeans.labels_)
        silhouette_scores.append((score, n_clusters))

    # Select the number of clusters with the highest silhouette score
    return max(silhouette_scores)[1]

def perform_clustering(vectors, method):
    embeddings = np.array(list(vectors.values())).reshape(-1, 1)
    if method == 'kmeans':
        optimal_clusters = find_optimal_clusters(embeddings)
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=0).fit(embeddings)
        labels = kmeans.labels_
    elif method == 'affinity':
        affinity = AffinityPropagation(random_state=0).fit(embeddings)
        labels = affinity.labels_
    elif method == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5).fit(embeddings)
        labels = clusterer.labels_
    else:
        raise ValueError("Invalid clustering method specified")

    clusters = {}
    for key, label in zip(vectors.keys(), labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(key)
    return clusters

def write_clusters_to_file(clusters, output_file, facet_num):
    with open(output_file, 'a') as file:
        file.write(f"Facet {facet_num}:\n")
        for i, cluster in clusters.items():
            file.write(f"Cluster {i + 1}: {', '.join(cluster)}\n")
        file.write("\n")

def main(centroids_file, con_emb_file, output_file, method):
    centroids = load_centroids(centroids_file)
    concepts_embeddings = load_embeddings(con_emb_file)

    for i, centroid in enumerate(centroids):
        vectors = calculate_vectors(centroid.numpy(), concepts_embeddings)
        clusters = perform_clustering(vectors, method)
        write_clusters_to_file(clusters, output_file, i + 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster normalized vectors and output the results.")
    parser.add_argument('--centroids', type=str, required=True, help='Path to input .pt file with centroids')
    parser.add_argument('--con_embeddings', type=str, required=True, help='Path to input .pkl file with concepts embeddings')
    parser.add_argument('--output', type=str, required=True, help='Path to output .txt file for clusters')
    parser.add_argument('--method', type=str, choices=['kmeans', 'affinity', 'hdbscan'], default='kmeans', help='Clustering method: kmeans, affinity, or hdbscan')

    args = parser.parse_args()

    main(args.centroids, args.con_embeddings, args.output, args.method)

