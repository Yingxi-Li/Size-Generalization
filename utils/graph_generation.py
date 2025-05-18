import networkx as nx 
import numpy as np 
import matplotlib.pyplot as plt

def generate_graph(n, p, seed=None, weighted=True):
    G = nx.fast_gnp_random_graph(n, p, seed=seed)
    # G = nx.barabasi_albert_graph(n, 4, seed=seed)
    
    if weighted:
        weight = {}
        for e in G.edges():
            weight[e] = np.random.rand()
        nx.set_edge_attributes(G, weight, "weight")
    
    return G 

def generate_durer(n):
    G = nx.Graph()
    G.add_nodes_from(range(2*n))
    
    for i in range(n):
        if i + 2 < n:
            G.add_edge(i, i+2)
        else:
            G.add_edge(i, i+2-n)
        outer_node_idx = i + n
        G.add_edge(i, outer_node_idx)
        if outer_node_idx + 1 < 2*n:
            G.add_edge(outer_node_idx, outer_node_idx+1)
        else:
            G.add_edge(outer_node_idx, outer_node_idx-n+1)
    
    print("edges:", G.edges())
    return G
    

def sample_one_small_graph(G, graph_size, seed):
    """Sample a small graph from a large graph by selecting vertices uniformly at random.

    Args:
        G: Large graph to select from. 
        graph_size: Size of the small graph, need to be less than size of the big graph.
        seed: random seed. 

    Returns:
        _type_: networkx Graph. 
    """
    # np.random.seed(seed)
    selected_nodes = np.random.choice(G.nodes, size=graph_size, replace=False)
    # print(selected_nodes)
    subgraph = G.subgraph(selected_nodes)
    return subgraph

def sample_small_graphs(G, num_graphs, graph_size):
    graphs = []
    for i in range(num_graphs):
        graphs.append(sample_one_small_graph(G, graph_size, seed=i))
    return graphs

def relabel_graph_nodes(G):
    # Create a mapping from the original node labels to consecutive integers
    mapping = {node: i for i, node in enumerate(G.nodes)}

    # Use the mapping to relabel the nodes
    relabeled_G = nx.relabel_nodes(G, mapping)

    return relabeled_G

def complete_minus_two(n):
    G = nx.complete_graph(n)
    G.remove_edge(0, 1)
    G.remove_edge(2, 3)
    
    return G

if __name__ == "__main__":
    # G = generate_durer(7)
    nodes = [0, 1, 2, 3, 4, 5]
    edges = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (4, 5)]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.draw(G, with_labels = "True")
    plt.show()