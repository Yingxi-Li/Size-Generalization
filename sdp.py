import cvxpy as cp 
import numpy as np
import networkx as nx
from scipy.linalg import sqrtm, null_space
from numpy.linalg import matrix_rank

from utils.graph_generation import generate_graph, sample_one_small_graph, generate_durer, sample_one_small_graph, complete_minus_two
from utils.ip_solver import max_cut_ip
from utils.sparsification import sparsify_graph

from decimal import Decimal, getcontext

import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
    
    
def sdp_laplace(G, laplacian=None):
    if laplacian is None:
        laplacian = np.array(0.25 * nx.laplacian_matrix(G).todense())
        
    psd_mat = cp.Variable(laplacian.shape, PSD=True)
    obj = cp.Maximize(cp.trace(laplacian @ psd_mat))
    constraints = [cp.diag(psd_mat) == 1]  # unit norm
    constraints += [psd_mat >> 0]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    return prob.value, psd_mat.value

def sdp_dual(G):

    laplacian = np.array(0.25 * nx.laplacian_matrix(G).todense())
    n = G.number_of_nodes()
        
    y = cp.Variable(n)
    obj = cp.Minimize(sum(y))
    Y = cp.diag(y)
    constraint = [Y - laplacian >> 0]
    prob = cp.Problem(obj, constraint)
    prob.solve()
    
    return prob.value, y.value

def sdp_dual_laplace(laplacian):
    n = len(laplacian)

    y = cp.Variable(n)
    obj = cp.Minimize(sum(y))
    Y = cp.diag(y)

    constraint = [Y - laplacian >> 0]
    prob = cp.Problem(obj, constraint)
    prob.solve()
    
    return prob.value, y.value


def sdp_triangle(G, laplacian=None):
    """
    SDP relaxation of Max-Cut with the triangle constraints
    """
    if laplacian is None:
        laplacian = np.array(0.25 * nx.laplacian_matrix(G).todense())
    # print(nx.laplacian_matrix(G).todense())
        
    # Setup and solve the GW semidefinite programming problem
    psd_mat = cp.Variable(laplacian.shape, PSD=True)
    obj = cp.Maximize(cp.trace(laplacian @ psd_mat))
    constraints = [cp.diag(psd_mat) == 1]  # unit norm
    constraints += [psd_mat >> 0]
    for i in range(G.number_of_nodes()):
        for j in range(G.number_of_nodes()):
            for k in range(G.number_of_nodes()):
                if i != j and j != k and i != k:
                    constraints += [psd_mat[i, j] + psd_mat[j, k] + psd_mat[i, k] >= -1]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return prob.value, psd_mat.value


def goeman_williamson(G, X_val = None):
    if X_val is None:
        cut_weight, X_val = sdp_laplace(G)
    V = sqrtm(X_val)
    
    u = np.random.rand(G.number_of_nodes())
    x = np.sign(V @ u)
    
    left = []
    for i, node in enumerate(G.nodes()):
        if x[i] == 1:
            left.append(node)
            
    cut_weight = nx.cut_size(G, left, weight="weight")
    return cut_weight


def max_qp(G):
    cut_weight, X_val = sdp_laplace(G)
    V = sqrtm(X_val)
        
    u = np.random.rand(G.number_of_nodes())
    z = V @ u / np.sqrt(4 * np.log(G.number_of_nodes()))
    
    x = []
    for zi in z:
        if zi > 1:
            zi = zi / np.abs(zi)
        unif = np.random.rand()
        x.append(1 if unif < (1 + zi)/2 else -1)
        
    left = []
    for i, node in enumerate(G.nodes()):
        if x[i] == 1:
            left.append(node)

    cut_weight = nx.cut_size(G, left, weight="weight")
    # print(cut_weight)
    
    return cut_weight
    
    

def is_pos_sem_def(x):
    if np.all(np.linalg.eigvals(x) >= 0):
        return True 
    else:
        return np.round(np.min(np.linalg.eigvals(x)), 3)
    
    
def convert_values(matrix):
    """
    Convert values > 1 to 1 and values < -1 to -1 in a NumPy matrix.
    
    Parameters:
        matrix (numpy.ndarray): Input matrix
    
    Returns:
        numpy.ndarray: Converted matrix
    """
    converted_matrix = np.where(matrix > 1, 1, matrix)
    converted_matrix = np.where(converted_matrix < -1, -1, converted_matrix)
    return converted_matrix

def mat_dot(A, B):
    return np.trace(A @ B)

def truncate(X, eps = 0.01):
    """
    Round all values of X to 10^-6. Then, truncate all numbers with magnitude 
    less than between 1-eps and 1 but not including 1 to 1 - eps. 
    """
    
    X = np.round(X, 6)
    X = np.where(np.abs(X) > 1 - eps and np.abs(X) != 1.0, np.sign(X) * (1 - eps), X)
    
def plot_G(G):
    """Plot networkX graph G with node labels. 
    Layout is plannar. """

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
        
def expected_unweighted_gw(G, X):
    res = 0
    for edge in G.edges():
        i, j = edge
        res += np.arccos(np.round(X[i, j], 3))/np.pi
    return res

if __name__ == "__main__":
    
    G = nx.complete_graph(4)
    sdp_val, X = sdp_laplace(G)
    print("SDP val:", sdp_val)
    print(X)
