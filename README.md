# Network and Clustering Portfolios on the FTSE 100

This project compares portfolio construction methods on FTSE 100 equities, examining whether network-based and clustering approaches that exploit correlation structures can deliver more robust out-of-sample risk–return profiles than classical Markowitz optimization.

## Overview

Using daily data from 2015–2025, we build portfolios calibrated on 2015–2017 and evaluate their performance over 2018–2025 against the FTSE 100 benchmark. The analysis includes:

- **Markowitz portfolios**: Minimum-variance and maximum-Sharpe as baseline methods
- **Network portfolios**: Degeneracy ordering, clique centrality, and eigenvector centrality portfolios built from thresholded correlation graphs
- **Clustering portfolios**: Cluster equal-weight, hierarchical 1/N, HRP, and HERC portfolios using hierarchical clustering

## Key Findings

The degeneracy ordering portfolio and cluster-equal portfolio outperform both Markowitz portfolios and the FTSE 100 index out-of-sample.

## Author

**Nikos Bellos**  
Member of AUEB Students' Investment Finance Club
