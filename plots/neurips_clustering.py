"""
For neurips submission, unified plot format for maxcut and clustering
"""

import numpy as np
import matplotlib.pyplot as plt

magnolia_colors = ['#ff1f5b',  '#009ade', '#af58ba', '#00cd6c']

data = {
    "GM": {
        "x"  : [0.2, 0.4, 0.6, 0.8],
        "SL": {
            "baseline": 0.502,                      
            "mean"    : [0.5760, 0.5300, 0.5220, 0.5095],
            "ci"      : [0.139943, 0.020125, 0.010770, 0.007228] / np.sqrt(100) * 1.96,
        },
        "k-means++": {
            "baseline": 0.914632,
            "mean"    : [0.905248, 0.907090, 0.916470, 0.909312],
            "ci"      : [0.107829, 0.108207, 0.097537, 0.105719] / np.sqrt(100) * 1.96,
        },
        "softmax k-centers": {
            "baseline": 0.882424,
            "mean"    : [0.882266, 0.873638, 0.878878, 0.880936],
            "ci"      : [0.128776, 0.134190, 0.131322, 0.125343]/ np.sqrt(100) * 1.96,
        },
    },

    "NC": {
        "x"  : [0.2, 0.4, 0.6, 0.8],
        "SL": {
            "baseline": 1.0,
            "mean"    : [0.7400, 0.9290, 1.0000, 1.0000],
            "ci"      : [0.172858, 0.142281, 0.000000, 0.000000]/ np.sqrt(100) * 1.96,
        },
        "k-means++": {
            "baseline": 0.620462,
            "mean"    : [0.623460, 0.625480, 0.622324, 0.625704],
            "ci"      : [0.064566, 0.065145, 0.065517, 0.064743]/ np.sqrt(100) * 1.96,
        },
        "softmax k-centers": {
            "baseline": 0.612822,
            "mean"    : [0.616966, 0.615158, 0.617446, 0.614806],
            "ci"      : [0.064961, 0.065444, 0.064758, 0.065498]/ np.sqrt(100) * 1.96,
        },
    },

    "MNIST": {
        "x"  : [0.2, 0.4, 0.6, 0.8, 1.0],
        "SL": {
            "baseline": 0.502,
            "mean"    : [0.5420, 0.5220, 0.5140, 0.5090, 0.5020],
            "ci"      : [0.030594, 0.014697, 0.016384, 0.008155, 0.000000]/ np.sqrt(100) * 1.96,
        },
        "k-means++": {
            "baseline": 0.669964,
            "mean"    : [0.666662, 0.670610, 0.671832, 0.661300, 0.664496],
            "ci"      : [0.099922, 0.101882, 0.099618, 0.101106, 0.099999]/ np.sqrt(100) * 1.96,
        },
        "softmax k-centers": {
            "baseline": 0.605225,
            "mean"    : [0.641175, 0.631175, 0.622850, 0.606675, 0.611025],
            "ci"      : [0.106285, 0.104061, 0.095879, 0.090808, 0.090645]/ np.sqrt(100) * 1.96,
        },
    },

    "OMNIGLOT": {
        "x"  : [0.2, 0.4, 0.6, 0.8, 1.0],
        "SL": {
            "baseline": 0.525000,
            "mean"    : [0.700000, 0.593750, 0.566667, 0.540625, 0.525000],
            "ci"      : [0.139194, 0.050389, 0.062361, 0.031406, 0.000000]/ np.sqrt(100) * 1.96,
        },
        "k-means++": {
            "baseline": 0.657675,
            "mean"    : [0.663100, 0.663650, 0.657650, 0.663700, 0.660600],
            "ci"      : [0.109452, 0.110005, 0.107012, 0.111293, 0.111019]/ np.sqrt(100) * 1.96,
        },
        "softmax k-centers": {
            "baseline": 0.605225,
            "mean"    : [0.641175, 0.631175, 0.622850, 0.606675, 0.611025],
            "ci"      : [0.106285, 0.104061, 0.095879, 0.090808, 0.090645]/ np.sqrt(100) * 1.96,
        },
    },
}

algo_order   = ["SL", "k-means++", "softmax k-centers"]
nice_colors  = ['#0072B2', '#D55E00', '#009E73']          
nice_colors = ['#009ade', '#bf212f', '#00cd6c']
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
    for algo, color in zip(algo_order, nice_colors):
        rec = dset[algo]
        # dotted baseline if we have one
        if rec["baseline"] is not None:
            ax.hlines(rec["baseline"], x[0], x[-1], ls=':', lw=1, color=color, label='subsampled '+algo)
        # mean curve
        ax.plot(x, rec["mean"], marker='o', lw=1.6, color=color, label=algo)
        # 95 % CI ribbon (assuming ci == ±1 sd)
        lo = np.array(rec["mean"]) - np.array(rec["ci"])
        hi = np.array(rec["mean"]) + np.array(rec["ci"])
        ax.fill_between(x, lo, hi, alpha=0.25, color=color)
    ax.set_xticks(x)
    ax.tick_params(axis='x', rotation=35)

fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
fig.subplots_adjust(left=0.05, right=0.78,  # room for legend (unchanged)
                    top=0.93, bottom=0.12,  # tighten vertical margins
                    wspace=0.15)

for ax, title in zip(axes, titles):
    draw_panel(ax, data[title])
    ax.set_title(title, pad=6
                 )

axes[0].set_ylabel('Accuracy',
                   )
fig.supxlabel('Fraction of points subsampled'
              , y=0.02)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='center left',          
           bbox_to_anchor=(1.00, 0.5), 
           ncol=1,                     
           frameon=False)

# plt.show()
plt.savefig('clustering.pdf', bbox_inches='tight',   
            pad_inches=0.02)

