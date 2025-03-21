# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:50:06 2024

@author: uqhjian5
"""
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import torch

def calculate_symmetric_normalized_laplacian(adj: np.ndarray) -> np.matrix:
    """Calculate yymmetric normalized laplacian.
    Assuming unnormalized laplacian matrix is `L = D - A`,
    then symmetric normalized laplacian matrix is:
    `L^{Sym} =  D^-1/2 L D^-1/2 =  D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2`
    For node `i` and `j` where `i!=j`, L^{sym}_{ij} <=0.

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Symmetric normalized laplacian L^{Sym}
    """
    # add self loop
    adj = adj + np.diag(np.ones(adj.shape[0], dtype=np.float32))
    adj = sp.coo_matrix(adj)
    degree = np.array(adj.sum(1))
    # diagonals of D^{-1/2}
    degree_inv_sqrt = np.power(degree, -0.5).flatten()
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
    matrix_degree_inv_sqrt = sp.diags(degree_inv_sqrt)   # D^{-1/2}
    symmetric_normalized_laplacian = sp.eye(
        adj.shape[0]) - matrix_degree_inv_sqrt.dot(adj).dot(matrix_degree_inv_sqrt).tocoo()
    return symmetric_normalized_laplacian


def calculate_scaled_laplacian(adj: np.ndarray, lambda_max: int = 2, undirected: bool = True) -> np.matrix:
    """Re-scaled the eigenvalue to [-1, 1] by scaled the normalized laplacian matrix for chebyshev pol.
    According to `2017 ICLR GCN`, the lambda max is set to 2, and the graph is set to undirected.
    Note that rescale the laplacian matrix is equal to rescale the eigenvalue matrix.
    `L_{scaled} = (2 / lambda_max * L) - I`

    Args:
        adj (np.ndarray): Adjacent matrix A
        lambda_max (int, optional): Defaults to 2.
        undirected (bool, optional): Defaults to True.

    Returns:
        np.matrix: The rescaled laplacian matrix.
    """

    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    laplacian_matrix = calculate_symmetric_normalized_laplacian(adj)
    if lambda_max is None:  # manually cal the max lambda
        lambda_max, _ = linalg.eigsh(laplacian_matrix, 1, which='LM')
        lambda_max = lambda_max[0]
    laplacian_matrix = sp.csr_matrix(laplacian_matrix)
    num_nodes, _ = laplacian_matrix.shape
    identity_matrix = sp.identity(
        num_nodes, format='csr', dtype=laplacian_matrix.dtype)
    laplacian_res = (2 / lambda_max * laplacian_matrix) - identity_matrix
    return laplacian_res


def calculate_symmetric_message_passing_adj(adj: np.ndarray) -> np.matrix:
    """Calculate the renormalized message passing adj in `GCN`.
    A = A + I
    return D^{-1/2} A D^{-1/2}

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Renormalized message passing adj in `GCN`.
    """

    # add self loop
    adj = adj + np.diag(np.ones(adj.shape[0], dtype=np.float32))
    # print("calculating the renormalized message passing adj, please ensure that self-loop has added to adj.")
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mp_adj = d_mat_inv_sqrt.dot(adj).transpose().dot(
        d_mat_inv_sqrt).astype(np.float32)
    return mp_adj


def calculate_transition_matrix(adj: np.ndarray) -> np.matrix:
    """Calculate the transition matrix `P` proposed in DCRNN and Graph WaveNet.
    P = D^{-1}A = A/rowsum(A)

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Transition matrix P
    """

    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(row_sum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    prob_matrix = d_mat.dot(adj).astype(np.float32).todense()
    return prob_matrix

def load_pkl(pickle_file: str) -> object:
    """Load pickle data.

    Args:
        pickle_file (str): file path

    Returns:
        object: loaded objected
    """

    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def load_adj(file_path: str, adj_type: str):
    """load adjacency matrix.

    Args:
        file_path (str): file path
        adj_type (str): adjacency matrix type

    Returns:
        list of numpy.matrix: list of preproceesed adjacency matrices
        np.ndarray: raw adjacency matrix
    """

    try:
        # METR and PEMS_BAY
        _, _, adj_mx = load_pkl(file_path)
        
    except ValueError:
        # PEMS04
        adj_mx = load_pkl(file_path)
        print("&&&&&&&save 1")
        np.savetxt("adj1.csv", adj_mx, delimiter=',')
    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [calculate_symmetric_message_passing_adj(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        adj = [calculate_transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [calculate_transition_matrix(adj_mx).T, calculate_transition_matrix(adj_mx.T).T]
    elif adj_type == "amc":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32),
               calculate_symmetric_normalized_laplacian(adj_mx).astype(np.float32).A,
               calculate_symmetric_message_passing_adj(adj_mx).astype(np.float32).A]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == "original":
        adj = [adj_mx]
    else:
        error = 0
        assert error, "adj type not defined"
        print("&&&&&&&save 2")
        np.savetxt("adj2.csv", adj[0], delimiter=',')
    return adj, adj_mx

def get_edge_index(adj_mx, device):
    edge_index = []
    edge_weight = []
    for i in range(adj_mx.shape[0]):
        for j in range(adj_mx.shape[1]):
            if adj_mx[i,j] != 0 :
                edge_index.append(torch.tensor([i,j]).reshape([2,1]))
                edge_weight.append(adj_mx[i,j])
    return torch.cat(edge_index, 1).to(device), torch.Tensor(edge_weight).to(device)


if __name__ == '__main__':

    DATASET_NAME = "METR-LA"
    
    adj_mx, adj = load_adj("../datasets/" + DATASET_NAME +
                         "/adj_mx.pkl", "doubletransition")
    edge_index, edge_weight = get_edge_index(adj_mx[0], "cuda:0")#.to("cuda:0")






