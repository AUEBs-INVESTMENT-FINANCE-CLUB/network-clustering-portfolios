# Network and Clustering Portfolios on FTSE 100

Portfolio construction using network theory and clustering on FTSE 100 equities (2015-2025).

## Portfolios

- **Degeneracy**: Selects weakly connected stocks
- **Inverse Eigenvector centrality**: Inverse weighting by network centrality
- **Hierarchical clustering**: Equal weight across correlation clusters
- **K-means**: Clusters by return/volatility
- **HERC**: Hierarchical Equal Risk Contribution

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

## Authors

Nikolaos Bellos  
AUEB Students' Investment and Finance Club