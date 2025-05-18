import matplotlib.pyplot as plt
import numpy as np 
import networkx as nx

from greedy import greedy1, greedy2
from utils.graph_generation import generate_graph, sample_one_small_graph, relabel_graph_nodes, generate_durer
from utils.ip_solver import max_cut_ip
from sdp import sdp2, sdp_laplace, goeman_williamson, sdp_dual, max_qp

import time 

def one_line_ci(x, y, ci, x_name, y_name, title, fig_name, do_save = False, baseline=None):
    """
    Plot a line graph with confidence interval.
    """

    fig = plt.figure()
    fig.set_figwidth(9)
    fig.set_figheight(6)

    plt.plot(x, y, 'o-')
    plt.fill_between(x, np.array(y) - np.array(ci), np.array(y) + np.array(ci), alpha=0.3)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    
    if baseline:
        plt.axhline(y=baseline)

    if do_save:
        plt.savefig("figures/" + fig_name)
        

def multiple_lines_ci(x, y, ci, line, colors, label, fig_name, do_save=False, baseline=None):
    # Set font family for x-name, y-name, legend, and title
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.edgecolor'] = 'darkgray'
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    
    # plt.rcParams['legend.fontsize'] = 12

    fig = plt.figure()
    fig.set_figwidth(6.5)
    fig.set_figheight(5)
    
    for i in range(len(y)):
        print(x, y[i])
        plt.plot(x, y[i], line[i], c=colors[i], label=label[i])
        plt.fill_between(x, np.array(y[i]) - np.array(ci[i]), np.array(y[i]) + np.array(ci[i]), color=colors[i], alpha=0.2)
 
    if do_save:
        plt.savefig("figures/neurips_v2/" + fig_name)

def run_multiple_sample_sizes(sample_sizes, num_trails, graph_size):
    """
    Run greedy algorithm for multiple sample sizes and plot the results.
    """
    
    x = sample_sizes
    y = []
    ci = []
    G = generate_graph(graph_size, 0.8, seed=1, weighted=True)
    for sample_size in sample_sizes:
        weights = []
        for i in range(num_trails):
            sample_G = sample_one_small_graph(G, int(sample_size * G.number_of_nodes()), i)
            weight, _ = sdp_laplace(sample_G)
            weights.append(weight) 
        y.append(np.mean(weights))
        ci.append(1.96 * np.std(weights) / np.sqrt(len(weights)))
        
    x_axis = "Percentage of Nodes Sampled"
    y_axis = "Max Cut Density" 
    one_line_ci(x, y, ci, x_name = x_axis, y_name = y_axis, 
                title = "Generalization Performance of SGD, 100 Nodes, 0.8 Edge Density", 
                fig_name = "mc_density_sgd.pdf", 
                do_save = True, 
                baseline = y[-1])
    
    
def run_multiple_sizes_algorithms(sample_sizes, num_trails, graph_size):
    x = sample_sizes
    y = [[], []]
    ci = [[], []]
    G = generate_graph(graph_size, 0.8, seed=1, weighted=True)
    for sample_size in sample_sizes:
        weights1 = []
        weights2 = []
        for _ in range(num_trails):
            _, weight1, _ = greedy1(G, percent_sampled = sample_size)
            _, weight2, _ = greedy2(G, percent_sampled = sample_size)
            weights1.append(weight1) 
            weights2.append(weight2)
        y[0].append(np.mean(weights1))
        y[1].append(np.mean(weights2))
        ci[0].append(1.96 * np.std(weights1) / np.sqrt(len(weights1)))
        ci[1].append(1.96 * np.std(weights2) / np.sqrt(len(weights2)))
    
    _, exact_obj_value =  max_cut_ip(G)
        
    x_axis = "Percentage of Nodes Sampled"
    y_axis = "Max Cut Density" 
    multiple_lines_ci(x, y, ci, labels = ["", "Greedy Algorithm 2"],
                      x_name = x_axis, y_name = y_axis, 
                      title = "Generalization Performances, 40 Nodes, 0.8 Edge Density", 
                      fig_name = "mc_density_algs2.pdf", 
                      do_save = True, 
                      baseline = exact_obj_value)
    
def run_times_algorithms(sizes, num_trails):
    avg_times = []
    avg_times_ip = []
    
    ci = [[], []]
    for size in sizes:
        times = []
        ip_times = []
        for _ in range(num_trails):
            G = generate_graph(size, 0.8, seed=1, weighted=True)
            
            start_time = time.time()
            _, weight1, _ = greedy1(G, percent_sampled = 0.3)
            run_time = time.time() - start_time 
            
            start_time = time.time()
            max_cut_ip(G)
            run_time_ip = time.time() - start_time 
            
            times.append(run_time)
            ip_times.append(run_time_ip)
        avg_times.append(np.mean(times))
        avg_times_ip.append(np.mean(ip_times))  
        
        ci[0].append(1.96 * np.std(times) / np.sqrt(len(times)))
        ci[1].append(1.96 * np.std(ip_times) / np.sqrt(len(ip_times)))
    
    multiple_lines_ci(sizes, [avg_times, avg_times_ip], ci, labels = ["Greedy Algorithm", "IP"],
                      x_name = "Number of Nodes", y_name = "Solve Time", 
                      title = "Generalization Performances, 40 Nodes, 0.8 Edge Density", 
                      fig_name = "run_time.pdf", 
                      do_save = True)
    
    
def run_generalization(G, sampled_percents, sample_size, algorithm = "naive_greedy"):
    x = sampled_percents
    y = []
    ci = []
    
    sdp_G = sdp_laplace(G)[0]
    W = G.size(weight="weight")
    
    OPT, _ = None, None 
    
    def denom(G):
        W = nx.get_edge_attributes(G, "weight")
        if W == {}:
            denom = G.number_of_edges()
        else:
            denom = sum(W.values())
        return denom 
    
    n = G.number_of_nodes()
    for percent in sampled_percents:
        print("Running sample size: ", percent)
        densities = []
        if percent == 1 and algorithm == "sdp":
            sample_size = 1
        for i in range(sample_size):
            density = None
            small_G = sample_one_small_graph(G, int(percent * G.number_of_nodes()), i)
            t = small_G.number_of_nodes()
            if algorithm == "Greedy":
                _, weight = naive_greedy(small_G, num_rand_init = 50)
                if denom(small_G) == 0:
                    print("Subsampled graph has no edge.")
                else:
                    density = weight / t**2
            elif algorithm == "sdp":
                weight, _ = sdp_laplace(small_G)
                if denom(small_G) == 0:
                    print("Subsampled graph has no edge.")
                    break
                density = weight / t**2
            elif algorithm == "SDP_dual":
                weight, _ = sdp_dual(small_G)
                density = weight
            elif algorithm == "GW":
                weight = goeman_williamson(small_G)
                if denom(small_G) == 0:
                    print("Subsampled graph has no edge.")
                    break
                density = weight / t**2
            elif algorithm == "IP":
                small_G_modified = relabel_graph_nodes(small_G)
                _, weight = max_cut_ip(small_G_modified)
                density = weight / denom(small_G_modified)
            elif algorithm == "max_qp":
                weight = max_qp(small_G)
                density = weight / denom(small_G)
            else:
                raise ValueError(f"Algorithm {algorithm} not implemented.")
            if density:
                densities.append(density) 
        y.append(np.mean(densities))
        ci.append(1.96 * np.std(densities) / np.sqrt(len(densities)))
    return y, ci
    
def run_multiple_algorithms(G, sample_percents, num_trails, algs, fig_name):
    ys, cis, baselines = [], [], []

    for alg in algs:
        print("Running algorithm: ", alg)
        y, ci = run_generalization(G, sample_percents, num_trails, algorithm = alg)
        ys.append(y)
        cis.append(ci)
        baselines.append(y[-1])
    
    multiple_lines_ci(x = sample_percents, 
                      y = ys, 
                      ci = cis, 
                      labels = algs, 
                      x_name = None, 
                      y_name = None, 
                      title = "Erdos-Renyi", 
                      fig_name = fig_name, 
                      do_save = True, 
                      baseline = baselines)
    

def run_graph_densities(n, t, graph_dens, num_trails):
    OPT_ratios = []
    CIs = []
    for p in graph_dens:
        G = generate_graph(n, p, seed=1, weighted=False)
        OPT, _ = sdp_dual(G)
        weights = []
        for i in range(num_trails):
            small_G = sample_one_small_graph(G, t, seed = i)
            weight, _ = sdp_laplace(small_G)
            weights.append(weight)
        OPT_ratios.append(np.mean(weights) / (OPT * t * (t-1) / (n * (n-1))))
        CIs.append(1.96 * np.std(weights/(OPT*t*(t-1) / (n*(n-1)))) / np.sqrt(len(weights)))
    one_line_ci(x = graph_dens, 
                y = OPT_ratios, 
                ci = CIs, 
                x_name = "Graph Density", 
                y_name = "Partial Graph Density / Entire Graph Density", 
                title = "Density Ratio vs Graph Density", 
                fig_name = "complete_minus_1.pdf", 
                do_save = True, 
                baseline = 1.031579)

if __name__ == "__main__":
    
    num_trail = 100
    n = np.arange(20, 101, 10)
    densities = np.arange(0.1, 0.91, 0.1)

    for i in n:
        for density in densities:
            run_multiple_algorithms(nx.erdos_renyi_graph(i, density, seed=1), 
                sample_percents = [0.2, 0.4, 0.6, 0.8, 1], 
                num_trails = num_trail, 
                algs = ['GW', "Greedy"],
                fig_name = f"good_examples/erdos_renyi_{i}_{density}.pdf")
    