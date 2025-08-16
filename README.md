# STEP
Spatio-Temporal Expression Prediction

![Pipeline Overview](pipeline_overview_STEPmi_STEPmr.png)

## Overview
The goal of this project is to use neural network-based spatio-temporal graph models to predict temporal gene expression. We created two algorithms to predict mRNA and miRNA expression across time, STEPmr and STEPmi, by integrating Hi-C datasets capturing genome-wide gene interactions. With correlations of 77% for mRNA and 93% for miRNA, our models perform effectively and account for an important portion of the variance. Highly predictable genes are generally structural or regulatory genes involved in essential biological processes, whereas genes with context-dependent functions are less predictable. These results highlight the influence of interacting gene dynamics on temporal expression patterns.

## Peroblem Formulation
We aim to predict temporal gene expression for mRNAs and miRNAs while understanding the underlying biological context. Our goals are:
1. **Improve gene expression prediction by incorporating gene interactions and genome spatial organization from Hi-C data.
2. **Analyze predicted patterns biologically, using pathway and functional enrichment analyses.
