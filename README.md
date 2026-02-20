# Network and Clustering Portfolios on FTSE 100

Portfolio construction using network theory and clustering on FTSE 100 equities.

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

## Data
Place `bloomberg_prices.csv` in the project root (columns: `Date`, stock tickers, optional `FTSE Index`). Training and test windows are set in `config.py` (default: 2016–2018 in-sample, 2019–2025 out-of-sample).

## Run
```bash
python main.py
```
Results are written to `outputs/` (metrics CSVs, appendix weights table, correlation heatmap, network plots, dendrograms, K-means scatter, cumulative return charts).

## Authors

Nikolaos Bellos  
AUEB Students' Investment and Finance Club