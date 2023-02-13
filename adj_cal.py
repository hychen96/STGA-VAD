import numpy as np
from random import choice
import os
import pickle
from math import exp


def compress_feature(input_feat, output_size, global_size):
    # sample vertex
    if input_feat.shape[0] > output_size:
        step = input_feat.shape[0] // global_size
        start_point = np.random.randint(step)
        sample_index = np.linspace(start_point, input_feat.shape[0], global_size + 1, endpoint=False,
                                   dtype=int).tolist()
        local_size = output_size - global_size
        local_center = choice(sample_index)
        for i in range(local_center - local_size // 2, local_center + local_size // 2 + 1):
            if i < 0 or i >= input_feat.shape[0] or i in sample_index:
                continue
            sample_index.append(i)
    else:
        sample_index = np.arange(input_feat.shape[0], dtype=int).tolist()
    output_dimension = len(sample_index)

    # establish the adjacent matrix A^tilde
    adj = np.zeros((output_dimension, output_dimension))
    for i in range(output_dimension):
        for j in range(output_dimension):
            if i == j:
                adj[i][j] = 1.0
            else:
                adj[i][j] = np.exp(-abs(sample_index[i] - sample_index[j]))
    identity = np.identity(adj.shape[0])
    adj = adj + identity

    # compute the degree matrix D^tilde
    d = np.zeros_like(adj)
    for i in range(output_dimension):
        for j in range(output_dimension):
            d[i][i] += adj[i][j]
    
    # calculate the normalized adjacency A^hat
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    adj_hat = np.dot(np.dot(d_inv_sqrt, adj), d_inv_sqrt)

    # obtain the features of samples
    output_feat = np.zeros((output_dimension, input_feat.shape[-1]))
    for i in range(output_dimension):
        output_feat[i] = input_feat[sample_index[i]]

    return output_feat.astype(np.float32), adj_hat.astype(np.float32)


def graph_generator(raw_feat, output_size=32000, global_size=16000):  # raw_feat.shape: (l,4096)
    # L2-normalization
    feat = raw_feat / np.linalg.norm(raw_feat, ord=2, axis=-1).reshape(-1, 1)
    # Compress into 32 segments
    return compress_feature(feat, output_size, global_size)
