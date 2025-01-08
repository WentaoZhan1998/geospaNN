---
title: 'GeospaNN: An python package for geospatial neural networks'
tags:
  - Python
  - Pytorch
  - Graph neural networks
  - Geospatial data
  - Gaussian Process
  - Kriging
authors:
  - name: Wentao Zhan
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Abhirup Datta
    affiliation: 1
affiliations:
 - name: Department of Biostatistics, Johns Hopkins Bloomberg School of Public Health
   index: 1
date: 13 December 2024
bibliography: paper.bib
editor_options: 
  markdown: 
    wrap: 72
---

# Summary

In geographical science, datasets with spatial information are the . In
geostatistics, spatial linear mixed model (SPLMM)
$Y = X\beta + \epsilon(s)$ has been a standard model to account for the
fixed effect between observations $y$ and covariates $x$ as well as the
spatial effect $\epsilon(s)$ among spatial locations $s$. Usually, the
spatial dependency embedded in $\epsilon(s)$ is modeled through a
Gaussian Process which brings parsimony and efficient solution.

Recently, with the increasingly complicated interaction among the
variables, machine learning techniques have been introduced to
geostatistics, extending SPLMM to a non-linear scenario by replacing
$X\beta$ with $m(X)$. However, few of them were originally designed for
the dependent data structure. Even for those adaptive modifications
accounting for dependency, the covariance structure significantly
restricts their scalability. Given these background, we introduce
geospaNN, which resolve the above concerns through NN-GLS, a novel
rendition of Neural Networks (NN) proposed in @zhan2024neural. GeospaNN
simultaneously achieves mean function estimation, prediction (together
with prediction interval). To achieve computational efficiency, geospaNN
utilized the computationally convenien Nearest Neighbor Gaussian Process
(NNGP) approximation [@datta2016nearest, @datta2022nearest]. GeospaNN is
also the first python package implements NNGP for scalable
covariance-matrix computation, which can benefit other geospatial
computation tool.

# Statement of need

GeospaNN is a Python package for geospatial analysis using NN-GLS, a
novel extension of neural networks that explicitly accounts for spatial
correlation in the data. The package implements NN-GLS using PyTorch, an
open-source library widely used for building machine learning models. As
was illustrated in @zhan2024neural, GeospaNN is equivalently a
geographically-informed Graph Neural Network (GNN), and is embedded into
the PyG (PyTorch Geometric) framework designed for efficient GNN tool on
irregular data structures like graphs. GeospaNN is primarily designed
for researchers and scientists within the fields of machine learning and
spatial statistics, but can be generally used for estimation and
prediction tasks on any data with dependency structure. Additionally,
geospaNN provides user-friendly wrappers for data simulation,
preprocessing, and model training that simplifies the analytical
pipeline. Within geospaNN, the implementation of NNGP approximation is
of independent importance to NN-GLS, which enables scalable
covariance-matrix-inversion and relevant Python-based applications.

As the goal of geospaNN, we provide a light, efficient, friendly machine
learning tool for geospatial analysis. According to the simulation, for
data with half million observations, it takes less than an hour to run
the pipeline on a standard personal laptop. A significant part of
geospaNN has been used in articles such as @zhan2024neural,
@heaton2024adjusting. In the future, geospaNN will play an significant
role on the interface between machine learning and spatial statistics
and serve as a foundation to the scientific and methodological
explorations.

# The GeospaNN package

This section provides an overview of the package, including the NN-GLS
architecture and several technical details. The website of geospaNN is
also available [@geospaNN], providing get practical examples of using
geospaNN and detailed documentations. A vignette is provided at
[@geospaNN] to illustrate the typical usage of the package.

## NN-GLS overview

In a simple linear regression scenario, according to the Gauss-Markov's
Theorem, when the data has a correlation structure, generalized least
squares (GLS) is more efficient than ordinary least square (OLS). For
vanilla neural networks, people assume independent observations $Y_i$'s
and use mean squared error as the loss function for regression task.
$Y = m(X) + \epsilon$. The OLS vs GLS example motivated the introduction
of the GLS-style loss,
$$ L\big(\hat{m}(\cdot)\big) = \frac{1}{n}\big(Y - \hat{m}(X)\big)^{\top}\Sigma^{-1}\big(Y - \hat{m}(X)\big)$$.

However, there are several practical issues for minimizing the GLS-style
loss in practice.

1.  The covariance matrix $\Sigma$ comes from a parametric covariance
    function, which is unknown in practice.
2.  Even $\Sigma$ is well-estimated, when sample size $n$ goes large,
    inverting $\Sigma$ will be computationally infeasible.
3.  Since the GLS loss is not additive across observations,
    mini-batching, a key ingredient to the success of modern deep NN,
    will not be applicable.

NN-GLS addresses the issues all at once by introducing NNGP to derive an
approximation for $\Sigma^{-1}$ and naturally equating itself as a
special Graphical Neural Networks. In brief, NNGP is an
nearest-neighbor-based approximation to a full Gaussian Process with a
specific covariance structure. On the matrix level, given a covariance
matrix $\Sigma$, its inverse can be approximated by
$$\Sigma^{-1} = Q = Q^{\top/2}Q^{1/2}$$ where $Q^{1/2}$ is a lower
triangular sparse matrix where the $j$th element on $i$th row is
non-zero if and only if $j$ is in $i$'s $k$-nearest neighborhood, where
$k$ is a pre-specified neighbor size. $\Sigma^{-1}$ is the core to both
likelihood and GLS-style loss computation, thus significantly simplify
the maximum likelihood estimation for spatial parameters and $\Sigma$
(issue 1), as well as the loss-minimization for mean function $\hat{m}$
(issue 2 and 3). Specifically for the GLS-loss function:
$$L\big(\hat{m}(\cdot)\big) = \frac{1}{n}\big(Y - \hat{m}(X)\big)^{\top}\Sigma^{-1}\big(Y - \hat{m}(X)\big) = \frac{1}{n}\big(Y^* - \hat{m}^*(X)\big)^{\top}\big(Y^* - \hat{m}^*(X)\big)$$,
where $Y^* = Q^{1/2}Y$ and $\hat{m}^*(X) = Q^{1/2}\hat{m}(X)$. The
GLS-loss returns to an additive format, and mini-batching can be applied
instead of full-batch. In geospaNN, the BRISC R package is used for
likelihood-based parameter estimation.

In NN-GLS, we assume the covariance structure to be unknown, the spatial
parameters $\theta$ and the mean function $\hat{m}$ are estimated
iteratively. The model training proceeds until the validation loss
converges. Based on the estimations, nearest-neighbor-based kriging is
used for the prediction and confidence interval at new locations.

## NNGP and other features

Alongside the estimation and prediction tasks, geospaNN includes an
efficient implementation of NNGP approximation. Given an $n\times n$
covariance matrix $\Sigma$ and a k-neighbor list (nearest neighbors by
default), our implementation guarantees a $O(n)$ computational
complexity for any aproximate matrix products involving $\Sigma^{1/2}$,
$\Sigma^{-1/2}$, and $\Sigma^{-1}$. NNGP approximation is foundamental
to several important scalable features in geospaNN, including spatial
data simulation and kriging.

# Acknowledgements

This work is supported by National Institute of Environmental Health
Sciences grant R01ES033739. The authors report there are no competing
interests to declare.

# References
