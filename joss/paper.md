---
title: '**geospaNN**: A Python package for geospatial neural networks'
tags:
- Python
- Pytorch
- Graph neural networks
- Geospatial data
- Gaussian Process
- Kriging
output: pdf_document
authors:
- name: Wentao Zhan
  affiliation: 1
- name: Abhirup Datta
  affiliation: 1
bibliography: paper.bib
editor_options:
  markdown:
    wrap: 72
affiliations:
- name: Department of Biostatistics, Johns Hopkins Bloomberg School of Public Health
  index: 1
---

### Summary

Geostatistical models are widely used to analyse datasets with spatial information encountered frequently in the geosciences (climate, ecology, forestry, environmental sciences, and other research fields). In geostatistics, the spatial linear mixed effects model (SPLMM)\
$$
Y = X\beta + \epsilon(s)
$$
has long been the standard approach to model such data. SPLMM accounts for the fixed effects between observations $Y$ and covariates $X$ via the linear regression part, while accounting for the correlation across spatial locations $s$ via the spatial error process $\epsilon(s)$. Typically, the spatial dependency embedded in $\epsilon(s)$ is modeled using a Gaussian Process, which provides a parsimonious and flexible solution to modeling spatial correlations and provide spatial predictions at new locations via kriging. 

Recently, machine learning techniques like Neural Networks (NNs) have been increasingly incorporated into geostatistics to address complex and non-linear interactions among variables. In this paper, we introduce `geospaNN`, a Python software for analysis of geospatial data. `geospaNN` implements a novel adaptation of NN <!--proposed in @zhan2024neural that extends the SPLMM to non-linear scenarios while retaining the capacity to directly model spatial correlations. `geospaNN`--> and simultaneously performs non-linear mean function estimation using a NN and spatial predictions (including prediction intervals) using Gaussian processes. For computational efficiency, `geospaNN` leverages the widely popular Nearest Neighbor Gaussian Process (NNGP) approximation [@datta2016nearest, @finley2019efficient]. It is also, to our knowledge, the first Python package to implement scalable covariance matrix computations in geostatistical models. <!--This is a standalone component of broad and  independent utility. benefiting other geospatial computation tools. -->



### Statement of Need

`geospaNN` is a Python package for geospatial analysis that uses NN-GLS [@zhan2024neural], a novel and scalable class of NNs explicitly designed to account for spatial correlation in the data. The package is effectively represented as a geographically-informed Graph Neural Network (GNN) by using NNGP covariance matrices. It is embedded within the PyG framework, which is designed for scalable execution of GNNs on irregular data structures like graphs. `geospaNN` is primarily intended for researchers and scientists in machine learning and geostatistics, but can also be applied more generally to estimation and prediction tasks involving other data with dependency structures, like time-series. `geospaNN` provides lightweight, user-friendly wrappers for data simulation, preprocessing, and model training, which significantly simplify the analytical pipeline. A portion of `geospaNN` has already been used in articles such as @zhan2024neural and @heaton2024adjusting. In the future, we anticipate that `geospaNN` will play a significant role at the interface of machine learning and spatial statistics, serving as a foundation for both scientific and methodological explorations.

The implementation of NNGP models within `geospaNN` is of independent importance. NNGP enables scalable covariance matrix inversions, which features extensively in geospatial models. There exist two widely used R-packages `spNNGP` [@finley2019efficient] and `BRISC` [@saha2018brisc] for implementations of NNGP, but to our knowledge there is no Python implementation of NNGP. We thus offer an avenue to efficiently analyze massive geospatial datasets in Python. <!---using NNGP (linear models) and NN-GLS (non-linear models). based applications.-->

# State of the field
In Python, to integrate geospatial data with deep learning, several specialized tools have been developed. Notably, `TorchGeo` [@TorchGeo2022] extends `PyTorch` [@paszke2019pytorch] for tasks such as land cover classification, object detection, and geospatial segmentation. Independently, the R package `geodl` [@maxwell2024geodl] was recently introduced for analyzing geospatial and spatiotemporal data.  However, these tools primarily supports raster and vector data, such as satellite imagery, which limits their general applications.

GNNs are the most common approaches for data with irregular geometry. It efficiently processes graph-structured data by learning graph-level representations through message passing and aggregation. For GNNs implementation, the PyTorch Geometric (PyG) library, provides a highly customizable framework for defining graph convolutional layers [@PyG2019]. GNNs have been widely applied in geospatial analysis, including crop yield prediction [@fan2022gnn], and traffic flow modeling [@wang2020traffic]. However, despite their growing adoption, there is still no systematic analytical software tailored for broad use within the statistical community.

# The geospaNN Package

This section provides an overview of the `geospaNN` package, including the NN-GLS architecture and several technical details. For practical examples and detailed documentation, visit the [`geospaNN` website](https://wentaozhan1998.github.io/geospaNN-doc). A [vignette](https://github.com/WentaoZhan1998/geospaNN/blob/main/vignette/vignette.pdf) is also available for detailed illustration of the package.

## NN-GLS Overview
<!---
In a simple linear regression scenario, Gauss-Markov's Theorem states that when the data exhibits a correlation structure, generalized least
squares (GLS) provides greater efficiency than ordinary least squares (OLS). For vanilla neural networks, it is typically implicitly assumed that the
observations $Y_i$ are not correlated, and mean squared error is used as the loss function for regression tasks for continuous outcomes:
$$
Y = m(X) + \epsilon.
$$
-->

NNGLS considers a simple model for spatial data:
$$
Y(s) = m(X(s)) + \epsilon(s)
$$
where $Y(s)$ and $X(s)$ are respectively the outcome and covariates observed at location $s$, $m$ is a non-linear function relating $X(s)$ to $Y(s)$, to be estimated using a NN. The key distinction from the standard non-linear regression setup is that here the errors $\epsilon(s)$ is a dependent process that models spatial correlation. <!---Typically it is decomposed as the sum of a Gaussian process for the spatial dependence and, possibly, a noise component. Given the observations $Y$, the covariates $X$, and the locations $s$ as the input, NN-GLS estimates the mean function $m()$ and predicts $Y$ at new locations. The ultimate goal of NN-GLS is to help understanding the complex relationship between observations and covariates and provide accurate geospatial prediction.
-->

Let $\Sigma$ be a model for the spatial error $\epsilon$, then the well-known theory (Gauss-Markov theorem) of OLS and GLS from the statistical literature motivates the introduction of a GLS-style loss in NNs: 
$$
L\big(m(\cdot)\big) = \frac{1}{n}\big(Y - m(X)\big)^{\top}\Sigma^{-1}\big(Y - m(X)\big).
$$

However, minimizing this GLS-style loss in practice presents several challenges:

1.  The covariance matrix $\Sigma$ is based on a parametric covariance function, and the parameters are typically unknown.
2.  Even if $\Sigma$ is well-estimated, inverting $\Sigma$ becomes computationally infeasible as the sample size $n$ grows large.
3.  Since the GLS loss is not additive across observations, mini-batching---an essential technique used in implementation of modern NNs---cannot be applied directly.

NN-GLS addresses these issues by introducing the NNGP to approximate $\Sigma^{-1}$, and it naturally equates to a specialized GNN. In brief, NNGP is a nearest-neighbor-based approximation to a full Gaussian Process with a 
specific covariance structure [@datta2022nearest].<!--Mathematically, given a covariance matrix $\Sigma$, its inverse can be approximated as: 
$$
\Sigma^{-1} = Q = Q^{\top/2}Q^{1/2},
$$
where $Q^{1/2}$ is a lower triangular sparse matrix. The $j$-th element in the $i$-th row of $Q^{1/2}$ is non-zero if and only if location $j$ is in the $k$-nearest neighborhood of location $i$, where $k$ is a pre-specified neighborhood size. This approximation simplifies 
 On a finite sample,--> NNGP sparsifies the covariance matrix, thus simplifying both likelihood computation and the GLS-style loss, addressing the three issues mentioned above. Specifically, for the GLS loss function: 
$$
L\big(m(\cdot)\big) = \frac{1}{n}\big(Y - m(X)\big)^{\top}\Sigma^{-1}\big(Y - m(X)\big) = \frac{1}{n}\big(Y^* - m^*(X)\big)^{\top}\big(Y^* - m^*(X)\big) = \frac{1}{n}\sum_{i = 1}^n(Y^*_i - m^*(X_i))^2 ,
$$
where $Y^* = Q^{1/2}Y$ and $m^*(X) = Q^{1/2}m(X)$ can be obtained easily using aggregation over the nearest-neighbot directed acyclic graph specifying the NNGP approximation. The GLS loss returns to an additive form, allowing for mini-batching instead
of full-batch training. <!--For likelihood-based parameter estimation, 
`geospaNN` uses the `BRISC` R package as an efficient solution [@saha2018brisc].-->In NN-GLS, <!--we assume that the parameters of the covariance matrix is unknown. T--> the spatial parameters $\theta$ and the weights and biases parameters of the NN used to model $m$ are estimated iteratively, and training proceeds until the validation loss
converges. Once estimation is complete, nearest-neighbor-based kriging is used to generate spatial predictions at new
locations. The whole procedure embeds the three core features of `geospaNN`,

1.  estimate the non-linear mean function by $\hat{m}$.
2.  estimate the spatial parameters by $\hat{\theta}$.
3.  predict the outcome at new locations by $\hat{Y}$.

## NNGP and Other Features

In addition to estimation and prediction for spatial mixed models using NN-GLS, `geospaNN` offers a suite of additional features that support a wide range of geospatial analyses. <!--Given an $n \times n$ covariance matrix $\Sigma$ and a $k$-neighbor list (defaulting to nearest
neighbors), our implementation guarantees $O(n)$ computational
complexity for any approximate matrix products involving $\Sigma^{1/2}$,
$\Sigma^{-1/2}$, and $\Sigma^{-1}$. The option of customizing the $k$-neighbor list is provided in case specially designed neighbor set is desired, like using distant neighbors to model long-range correlation [@stein2004approximating], or neighbors based on geodesic-distances if working with data collected on a non-Euclidean domain like water bodies [@gilbert2024visibility]. NNGP is fundamental to several key
scalable features in `geospaNN`, including spatial data simulation and
kriging.--> `geospaNN` provides simulation module allowing users to customize the spatial parameters and mean functions to generate $Y$, $X$, and $s$. Users are allowed to customize the spatial coordinates to simulate under different context. <!--This feature can be flexibly used to simulate a Gaussian process as a general spatial random effect by specifying a zero mean function. For kriging, instead of using the full observed set, only the nearest neighbors of each new spot in the observed set are used to compute the kriging weights.--> `geospaNN` implements nearest neighbor kriging, an alternate to full kriging, which has been shown in @zhan2024neural to guarantee accurate prediction interval under various settings. For essential machine learning tasks, `geospaNN` offers modules including NN architecture design, training log report, and result visualization. <!--The training log visualizes the evolution of validation loss as well as spatial parameters, providing deeper insight into model convergence. To interpret high-dimensional result, `geospaNN` incorporates partial dependence plots (PDPs) to illustrate the marginal effect of individual covariates on the outcome. When compared across multiple models, PDPs offer invaluable insights into the complex relationships between covariates and the predicted outcome.--><!--As a special case of the spatial non-linear mixed model, SPLMM with NNGP covariance is also implemented in `geospaNN`. The implementation here is--> `geospaNN` also implements SPLMM solution as a special case of NN-GLS. It should be an optimal choice for the Python users if efficient SPLMM solution is wanted for large geospatial datasets.

All these functions are included explicitly in the package and can be called independently (see [vignette](https://github.com/WentaoZhan1998/geospaNN/blob/main/vignette/vignette.pdf)).

# Discussion

The `geospaNN` package offers an efficient implementation of NN-GLS
approach proposed in @zhan2024neural. NN-GLS embeds NN with the spatial mixed model, accounting for spatial
correlation by replacing the original loss function with a GLS-style
version. `geospaNN` is capable of performing various statistical tasks,
including non-linear mean-function estimation, covariance parameter estimation, spatial prediction with uncertainty quantification.  <!--The NNGP approximation, which is fundamental to the scalable implementation of `geospaNN`, is offered as a standalone functionaility in `geospaNN`.This to our knowledge is the only Python implementation of the widely popular NNGP model for scalable geostatistical analysis.--> Due to the sparsity
of the NNGP approximation, `geospaNN` is seamlessly integrated into
the framework of GNNs, opening up new
possibilities for a wide range of advanced neural 
architectures. A promising future direction for `geospaNN` is to evolve
into a general framework for geospatially-informed deep learning, where
graph-based message-passing (convolution) can occur multiple times, with
weights determined by spatial processes to maintain statistical
interpretability. We are also planning to explore the extension of
`geospaNN` towards other data types and distributions in the future.

# Acknowledgements

This work is supported by National Institute of Environmental Health
Sciences grant R01ES033739. The authors report there are no competing
interests to declare.

# References
