import cvxpy as cp 
import numpy as np
import networkx as nx 


def max_cut_ip(G):
    """
    Integer programming solver for the max cut problem.
    """
    X = {}
    Z = {}
    
    for node in G.nodes():
        X[node] = cp.Variable(boolean=True, name="x_{}".format(node))
    for edge in G.edges():
        Z[edge] = cp.Variable(boolean=True, name="z_{}".format(edge))
        
    W = nx.get_edge_attributes(G, "weight")
    
    if W == {}:
        obj = sum(Z[i, j] for i, j in G.edges())
    else:
        obj = sum(W[i, j] * Z[i, j] for i, j in G.edges())
    
    constraints = []
    for i, j in G.edges():
        constraints += [Z[i, j] <= X[i] + X[j]]
        constraints += [Z[i, j] <= 2 - X[i] - X[j]]
        
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.GUROBI)
    
    left = []
    for node in G.nodes():
        if X[node].value == True:
            left.append(node)
        
    return left, prob.value
    
        
        
def max_cut_qp(G):
    """
    Quadratic Integer programming solver for the max cut problem.
    
    Args:
        G (NetworkX Graph): weighted graph
    """
    nodes = G.nodes()
    X = {}
    Bin_Var = {}
    for node in nodes:
        Bin_Var[node] = cp.Variable(boolean=True)
        X[node] = cp.Variable(name="x_{}".format(node))
    W = nx.get_edge_attributes(G, "weight")
    
    obj = 1/4 * sum(W[i, j] * (1 - X[i] * X[j]) for i, j in G.edges())
    
    constraints = []
    for node in nodes:
        constraints += [X[node] == -1 + 2 * Bin_Var[node]]

    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.GUROBI)
