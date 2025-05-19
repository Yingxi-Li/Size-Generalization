import networkx as nx
import numpy as np
import itertools

from utils.graph_generation import generate_graph, sample_one_small_graph
from utils.ip_solver import max_cut_ip

def edge_label(G, node1, node2, W):
    """
    Return the edge label of the edge between node1 and node2.
    """
    try: 
        W[(node1, node2)]
        return (node1, node2)
    except:
        return (node2, node1)
    # if G.has_edge(node1, node2):
    #     return (node1, node2)
    # elif G.has_edge(node2, node1):
    #     return (node2, node1)
    # else:
    #     raise ValueError("node1 and node2 cannot be the same")

def choose_side(node, left, right, G, W):
    """ 
    Place node on the side of cut that maximizes the total weight of 
    the crossing edges. 
    
    Args:
        node: node to be placed
        left, right: lists of nodes on the left and right side of the cut
        G (NetworkX Graph): weighted graph
        W (dict): edge weights
    """

    neighbors = G.neighbors(node)
    left_count = 0
    right_count = 0
    for neighbor in neighbors:
        if neighbor in left:
            if W == {}:
                right_count += 1
            else:
                right_count += W[edge_label(G, node, neighbor, W)]
        elif neighbor in right:
            if W == {}:
                left_count += 1
            else:
                left_count += W[edge_label(G, node, neighbor, W)]
    if left_count > right_count or (left_count == right_count and np.random.rand() > 0.5):
        left.append(node)
    else:
        right.append(node)
    return left, right

def greedy1(G, percent_sampled = 0.3):
    """Simplest to analyze greedy algorithm for the max cut problem.

    Args:
        G (NetworkX Graph): weighted graph
    """
    n = G.number_of_nodes()
    sample_G = sample_one_small_graph(G, int(percent_sampled * n), 0)
    W = nx.get_edge_attributes(G, "weight")
    print(W)
    
    sampled_nodes = sample_G.nodes()
    print("Number of node sampled: ", len(sampled_nodes))
    sampled_nodes_set = set(sampled_nodes)
    other_nodes = [i for i in G.nodes() if not i in sampled_nodes_set]
    
    best_cut_weight = 0
    best_cut = []

    for i in range(len(sampled_nodes) - 1):
        for c in [c for c in itertools.combinations(sampled_nodes, i+1)]:
            left = list(c) 
            right = list([i for i in sampled_nodes if not i in c])
            np.random.shuffle(other_nodes)
            for node in other_nodes:
                left, right = choose_side(node, left, right, G, W)
            weight = nx.cut_size(G, left, right, weight="weight")
            if weight > best_cut_weight:
                best_cut = left 
                best_cut_weight = weight
    return best_cut, best_cut_weight, sampled_nodes

def greedy2(G, percent_sampled = 0.3):
    """Fastest greedy algorithm for the max cut problem.

    Args:
        G (NetworkX Graph): weighted graph
    """
    near_opt_cut, weight, S = greedy1(G, percent_sampled)
    
    other_nodes = [i for i in G.nodes() if not i in set(S)]
    W = nx.get_edge_attributes(G, "weight")
    
    left = near_opt_cut
    right = [i for i in G.nodes() if not i in set(left)]
    for node in other_nodes:
        if node in left:
            left.remove(node)
        else:
            right.remove(node)
        left, right = choose_side(node, left, right, G, W)
        
    best_cut_weight = nx.cut_size(G, left, right, weight="weight")
        
    return left, best_cut_weight, S

def sample_cut(G):
    """
    Sample a random cut of the graph.
    """
    left = []
    right = []
    for node in G.nodes():
        if np.random.rand() > 0.5:
            left.append(node)
        else:
            right.append(node)
    return left, right
    
    
def naive_greedy(G, num_rand_init = 10, verbose = False):
    """
    Naive greedy algorithm for the max cut problem.
    """
    
    best_cut_weight = 0
    best_cut = []
    W = nx.get_edge_attributes(G, "weight")
    
    for _ in range(num_rand_init):
        if verbose:
            print("---------------Random Initialization------------------")
        left, right = sample_cut(G)
        
        rand_order = np.random.permutation(G.nodes())
        
        for node in rand_order:
            if node in left:
                left.remove(node)
            else:
                right.remove(node)
            left, right = choose_side(node, left, right, G, W)
            if verbose:
                print("Current weight: ", nx.cut_size(G, left, right, weight="weight"))
            
        cut_weight = nx.cut_size(G, left, right, weight="weight")
        if cut_weight > best_cut_weight:
            best_cut = left 
            best_cut_weight = cut_weight
        if verbose:
            print("Updated best cut weight: ", best_cut_weight)
        
        return best_cut, best_cut_weight