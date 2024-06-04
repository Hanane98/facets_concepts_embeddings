import argparse
import pickle
import torch
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.preprocessing import normalize
import numpy as np
import os

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

def write_vectors_to_file(vectors, output_file, i):
    output_file_name = f"{output_file}_{i}.pkl"
    with open(output_file_name, 'wb') as file:
        pickle.dump(vectors, file)

def main(centroids_file, in_embs_file, output_file):
    centroids = load_centroids(centroids_file)
    in_embs_embeddings = load_embeddings(in_embs_file)

    for i, centroid in enumerate(centroids):
        vectors = calculate_vectors(centroid.numpy(), in_embs_embeddings)
        write_vectors_to_file(vectors, output_file, i) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster normalized vectors and output the results.")
    parser.add_argument('--centroids', type=str, required=True, help='Path to input .pt file with centroids')
    parser.add_argument('--in_embs', type=str, required=True, help='Path to input .pkl file with input embeddings')
    parser.add_argument('--output', type=str, required=True, help='Path to output file for masked_con_emb, the beginning of the name')

    args = parser.parse_args()

    main(args.centroids, args.in_embs, args.output)


