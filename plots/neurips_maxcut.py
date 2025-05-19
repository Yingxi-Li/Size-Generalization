"""
For neurips submission, unified plot format for maxcut and clustering
"""

import numpy as np
import matplotlib.pyplot as plt

magnolia_colors = ['#ff1f5b',  '#009ade', '#af58ba', '#00cd6c']

data = {
    "Erdos-Renyi": {
        "x"  : [0.2, 0.4, 0.6, 0.8, 1.0],
        "Greedy": {
            "baseline": 0.19453333333333334,        
            "mean"    : [0.21326666666666666, 0.20673333333333335, 0.19942222222222222, 0.19677083333333334, 0.19453333333333334],
            "ci"      : [0.0024270021648420802, 0.001158581871337797, 0.0007747286087586291, 0.0004275492445104795, 0.00018854676299803457],
        },
        "GW": {
            "baseline": 0.19379466666666667,
            "mean"    : [0.2107333333333333, 0.20055000000000003, 0.19771111111111112, 0.19592083333333335, 0.19379466666666667],
            "ci"      : [0.0026975415221463604, 0.0016572466778096468, 0.000948989277830675, 0.0005829531950273479, 0.0004231816535181674],
        }
    },

    "Barbell": {
        "x"  : [0.2, 0.4, 0.6, 0.8, 1.0],
        "Greedy": {
            "baseline": 0.12559999999999996,        
            "mean"    : [0.133, 0.12863333333333335, 0.1269851851851852, 0.12642916666666665, 0.12559999999999996],
            "ci"      : [0.0022232540715506786, 0.0009153508552179042, 0.0003498551279505122, 0.00017942026032423131, 4.4418171880010265e-18],
        },
        "GW": {
            "baseline": 0.124208,
            "mean"    : [0.12426666666666665, 0.12534999999999996, 0.12460740740740739, 0.12460833333333331, 0.124208],
            "ci"      : [0.0021969622556140093, 0.0013951611710957747, 0.0005503214392950565, 0.0004041545734731442, 0.00026130406236073313],
        }
    },

    "Random Geometric": {
        "x"  : [ 0.2, 0.4, 0.6, 0.8, 1.0],
        "Greedy": {
            "baseline": 0.2481413333333333,        
            "mean"    : [0.249, 0.24925000000000003, 0.24835555555555552, 0.2483708333333334, 0.24811733333333333],
            "ci"      : [0.000577008376600086, 0.00026031049835831765, 0.00026351811030394406, 0.00014875742682984635, 8.917455140050644e-05]
        },
        "GW": {
            "baseline": 0.2369973333333333,
            "mean"    : [0.2369973333333333, 0.2369973333333333, 0.2369973333333333, 0.2369973333333333, 0.2369973333333333] ,
            "ci"      : [0.0029588811979705614, 0.002375325693067826, 0.002680755821870876, 0.0023949868838064676, 0.0021133719484102937]
        }

    },
    
    "Barabasi-Albert": {
        "x"  : [ 0.2, 0.4, 0.6, 0.8, 1.0],
        "Greedy": {
            "baseline": 0.061474666666666684,        
            "mean"    : [0.0746, 0.07191666666666667, 0.0663925925925926, 0.06347499999999999, 0.061474666666666684],
            "ci"      : [0.00423259164946385, 0.0019737894555435275, 0.0011273957608132874, 0.0006668202106515171, 0.00010351879227642846],
        },
        "GW": {
            "baseline": 0.061647999999999994,
            "mean"    : [0.07233333333333333, 0.06924999999999999, 0.06595555555555555, 0.06304166666666668, 0.061647999999999994],
            "ci"      : [0.004251856657630527, 0.0017759073424277766, 0.001219258235362515, 0.000718542918638773, 0.0001894506255278372],
        }
    }
}

algo_order   = ["Greedy", "GW"]
# nice_colors  = ['#0072B2', '#D55E00', '#009E73']          # colour-blind friendly
nice_colors = ['#009ade', '#bf212f', '#00cd6c']
nice_colors = ['#009ade', '#bf212f', '#bf212f']
titles       = list(data.keys())

plt.rcParams.update({
    # base font
    'font.family'      : 'Times New Roman',
    'font.size'        : 16,   

    # axes
    'axes.titlesize'   : 18,      
    'axes.labelsize'   : 18,    
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.linestyle'   : '--',
    'grid.alpha'       : 0.25,

    # ticks
    'xtick.labelsize'  : 13,
    'ytick.labelsize'  : 13,

    # legend
    'legend.fontsize'  : 17,
})


def draw_panel(ax, dset):
    x = np.array(dset['x'])
    all_y = []
    for algo, color in zip(algo_order, nice_colors):
        rec = dset[algo]
        # dotted baseline if we have one
        if rec["baseline"] is not None:
          if algo == "0.878 SDP":
            ax.hlines(rec["baseline"], x[0], x[-1], ls=':', lw=1, color=color)
          else:
            ax.hlines(rec["baseline"], x[0], x[-1], ls=':', lw=1, color=color, label=algo)
          all_y.append(rec["baseline"])
        # mean curve
        if algo == "0.878 SDP":
          ax.plot(x, rec["mean"], marker='o', lw=1.6, color=color)
        elif algo == "SDP":
          ax.plot(x, rec["mean"], marker='o', lw=1.6, color=color, label='subsampled SDP and 0.878 SDP')
        else:
          ax.plot(x, rec["mean"], marker='o', lw=1.6, color=color, label='subsampled '+algo)
        lo = np.array(rec["mean"]) - np.array(rec["ci"])
        hi = np.array(rec["mean"]) + np.array(rec["ci"])
        ax.fill_between(x, lo, hi, alpha=0.25, color=color)
        all_y.extend(lo)
        all_y.extend(hi)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='x', rotation=35)

    y_min, y_max = min(all_y), max(all_y)
    padding = 0.05 * (y_max - y_min)
    ax.set_ylim(y_min - padding, y_max + padding)


fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
fig.subplots_adjust(left=0.05, right=0.78, 
                    top=0.93, bottom=0.12, 
                    wspace=0.15)

for ax, title in zip(axes, titles):
    draw_panel(ax, data[title])
    ax.set_title(title, pad=6)

axes[0].set_ylabel('Density')
fig.supxlabel('Fraction of nodes subsampled', y=0.02)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='center left',          
           bbox_to_anchor=(1.00, 0.5), 
           ncol=1,                     
           frameon=False)


plt.savefig('maxcut.pdf', bbox_inches='tight',   
            pad_inches=0.02)