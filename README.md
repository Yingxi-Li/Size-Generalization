# Size-Generalization

> Experiments accompanying **“Accelerating Data-Driven Algorithm Selection for Combinatorial Partitioning Problems.”**


## 1. Quick start

You can set up an isolated environment (Python ≥ 3.10) with: 
```bash
python -m venv .venv          # or: python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```
And install all runtime dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

We organize all max-cut algorithms' implementation in the `max-cut` folder and clustering algorithms' implementation in the `clustering` folder. We also include in the `plots` folder the exact plotting code used in our paper. 

We include all datasets we used for the clustering experiments under the `clustering` folder. All max-cut data are generated using NetworkX generators, which we include as part of our dependencies. 