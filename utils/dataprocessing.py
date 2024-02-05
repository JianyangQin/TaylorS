import numpy as np
import math
from utils.tools import *
from utils.timefeatures import time_features
import pandas as pd
import pickle
import scipy.sparse as sp
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from copy import deepcopy

def normalize_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return A_reg.astype(np.float32)

def load_adj(adj_filename, num_of_vertices, id_filename=None):

    if 'npy' in adj_filename:
        adj_mx = np.load(adj_filename)
        return adj_mx, None
    elif 'npz' in adj_filename:
        adj = np.load(adj_filename)
        A = adj['adjacency']
        distanceA = adj['distance']
    elif 'pkl' in adj_filename:
        try:
            with open(adj_filename, 'rb') as f:
                pickle_data = pickle.load(f)
        except UnicodeDecodeError as e:
            with open(adj_filename, 'rb') as f:
                pickle_data = pickle.load(f, encoding='latin1')
        except Exception as e:
            print('Unable to load data ', adj_filename, ':', e)
            raise
        if ('HZ-METRO' in adj_filename) or ('SH-METRO' in adj_filename):
            distanceA = pickle_data
        else:
            _, _, distanceA = pickle_data
        A = deepcopy(distanceA)
        np.fill_diagonal(A, 0.)
        A[A > 0.] = 1.
    else:
        import csv
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
        distanceA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(adj_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distanceA[id_dict[i], id_dict[j]] = distance
            return A, distanceA
        else:
            with open(adj_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distanceA[i, j] = distance

    A = A.astype(np.float32)
    distanceA = distanceA.astype(np.float32)
        
    return A, distanceA

def load_se(se_filename):
    with open(se_filename, mode='r') as f:
        lines = f.readlines()
        temp = lines[0].split(' ')
        num_vertex, dims = int(temp[0]), int(temp[1])
        SE = np.zeros((num_vertex, dims), dtype=np.float)
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index] = [float(ch) for ch in temp[1:]]

    return SE

def load_dtw(dtw_filename, sigma, thres):
    dist_matrix = np.load(dtw_filename)

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    sigma = sigma
    dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > thres] = 1

    dtw_matrix = normalize_adj(dtw_matrix)
    return dtw_matrix

def load_origin_dtw(dtw_filename):
    dtw_distance = np.load(dtw_filename)
    nth = np.sort(dtw_distance.reshape(-1))[
          int(np.log2(dtw_distance.shape[0]) * dtw_distance.shape[0]):
          int(np.log2(dtw_distance.shape[0]) * dtw_distance.shape[0]) + 1]  # NlogN edges
    dtw_matrix = np.zeros_like(dtw_distance)
    dtw_matrix[dtw_distance <= nth] = 1
    dtw_matrix = np.logical_or(dtw_matrix, dtw_matrix.T)
    return dtw_matrix

def load_spatial(sp_filename, num_of_vertices, sigma, thres):
    if 'pkl' in sp_filename:
        try:
            with open(sp_filename, 'rb') as f:
                pickle_data = pickle.load(f)
        except UnicodeDecodeError as e:
            with open(sp_filename, 'rb') as f:
                pickle_data = pickle.load(f, encoding='latin1')
        except Exception as e:
            print('Unable to load data ', sp_filename, ':', e)
            raise
        if ('HZ-METRO' in sp_filename) or ('SH-METRO' in sp_filename):
            sp_matrix = pickle_data
        else:
            _, _, sp_matrix = pickle_data
        sp_matrix[sp_matrix == 1.] = 0.
    elif 'npz' in sp_filename:
        sp_matrix = np.load(sp_filename)['distance']
    else:
        import csv
    
        with open(sp_filename) as fp:
            dist_matrix = np.zeros((num_of_vertices, num_of_vertices)) + np.float('inf')
            file = csv.reader(fp)
            for line in file:
                break
            for line in file:
                start = int(line[0])
                end = int(line[1])
                dist_matrix[start][end] = float(line[2])
                dist_matrix[end][start] = float(line[2])
    
            # normalization
            std = np.std(dist_matrix[dist_matrix != np.float('inf')])
            mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
            dist_matrix = (dist_matrix - mean) / std
            sigma = sigma
            sp_matrix = np.exp(- dist_matrix ** 2 / sigma ** 2)
            sp_matrix[sp_matrix < thres] = 0
    
            sp_matrix = normalize_adj(sp_matrix)

    sp_matrix = sp_matrix.astype(np.float32)

    return sp_matrix

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def laplacian(W):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    d = 1 / np.sqrt(d)
    D = sp.diags(d, 0)
    I = sp.identity(d.size, dtype=W.dtype)
    L = I - D * W * D
    return L

def largest_k_lamb(L, k):
    lamb, U = linalg.eigsh(L, k=k, which='LM')
    return (lamb, U)


def eigen_vector(adj_mx):
    adj, k = adj_mx
    L = laplacian(adj)
    eig = largest_k_lamb(L, k)
    return eig

def neighbors(adj_mx):
    sampled_nodes_number = int(math.log(adj_mx.shape[0], 2))
    graph = csr_matrix(adj_mx)
    dist_matrix = dijkstra(csgraph=graph)
    dist_matrix[dist_matrix == 0] = dist_matrix.max() + 10
    localadj = np.argpartition(dist_matrix, sampled_nodes_number, -1)[:, :sampled_nodes_number]
    return localadj

def preprocess_adj(adj_mx, adj_type):
    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adj_type == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adj_type == "transition":
        adj = [asym_adj(adj_mx)]
    elif adj_type == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == 'eigen':
        adj = [eigen_vector(adj_mx)]
    elif adj_type == 'neighbors':
        adj = [neighbors(adj_mx)]
    else:
        raise ValueError('Fail to preprocess adjacent matrix')
    return adj