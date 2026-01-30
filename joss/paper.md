---
title: 'geospaNN: A Python package for geospatial neural networks'
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
  affiliation: 2
bibliography: paper.bib
editor_options:
  markdown:
    wrap: 72
affiliations:
- name: Department of Statistics, University of Wisconsin-Madison
  index: 1
- name: Department of Biostatistics, Johns Hopkins Bloomberg School of Public Health
  index: 2
---

### Summary
<!--
Geostatistical models are widely used to analyse datasets with spatial information encountered frequently in the geosciences (climate, ecology, forestry, environmental sciences, and other research fields). In geostatistics, the spatial linear mixed effects model (SPLMM)\
$$
Y = X\beta + \epsilon(s)
$$
has long been the standard approach to model such data. SPLMM accounts for the fixed effects between observations $Y$ and covariates $X$ via the linear regression part, while accounting for the correlation across spatial locations $s$ via the spatial error process $\epsilon(s)$. Typically, the spatial dependency embedded in $\epsilon(s)$ is modeled using a Gaussian Process, which provides a parsimonious and flexible solution to modeling spatial correlations and provide spatial predictions at new locations via kriging. 

Recently, machine learning techniques like Neural Networks (NNs) have been increasingly incorporated into geostatistics to address complex and non-linear interactions among variables. In this paper, we introduce `geospaNN`, a Python software for analysis of geospatial data. `geospaNN` implements a novel adaptation of NN <!--proposed in @zhan2024neural that extends the SPLMM to non-linear scenarios while retaining the capacity to directly model spatial correlations. `geospaNN`--> <!--and simultaneously performs non-linear mean function estimation using a NN and spatial predictions (including prediction intervals) using Gaussian processes. For computational efficiency, `geospaNN` leverages the widely popular Nearest Neighbor Gaussian Process (NNGP) approximation [@datta2016nearest, @finley2019efficient]. It is also, to our knowledge, the first Python package to implement scalable covariance matrix computations in geostatistical models. <!--This is a standalone component of broad and  independent utility. benefiting other geospatial computation tools. -->

Geostatistical models are essential for analyzing data with spatial structure across the geosciences, such as climate, ecology, and environmental science. At the same time, modern machine learning methods, especially neural networks (NNs), offer powerful tools for capturing complex, nonlinear relationships. Our package `geospaNN` bridges these two worlds by providing a Python library that integrates NN modeling with scalable spatial statistics. The software enables users to fit flexible spatial regression models, estimate complex mean structures, and generate Gaussian process (GP)–based spatial predictions with uncertainty quantification. Built on the `PyG` library designed for efficient graph neural network (GNN) training, `geospaNN` supports efficient computation on large, irregular spatial datasets. To handle modern geospatial data sizes, `geospaNN` incorporates the Nearest Neighbor Gaussian Process (NNGP) approximation [@datta2016nearest] for fast covariance computations. <!--To our knowledge, `geospaNN` is also the first Python package to offer scalable covariance matrix operations within a geostatistical modeling workflow, broadening the accessibility of large-scale spatial analysis.-->

### Statement of Need
Researchers in geoscience and related fields frequently need to model relationships among spatially distributed variables and generate reliable spatial predictions. Although many Python machine learning libraries can fit complex nonlinear regression models, they typically ignore spatial correlation, leading to biased estimates and misleading inference when applied to geospatial data. Existing spatial modeling tools in Python provide only partial solutions: some rely on complex neural architectures that sacrifice scientific interpretability, while others use full GP models whose computational demands make them impractical for large datasets.

`geospaNN` addresses these limitations by providing a spatial regression framework that combines the flexibility of NNs with the interpretability and statistical rigor of geostatistical models. It is designed for geoscientists, environmental researchers, and machine learning practitioners who need scalable and principled spatial modeling tools in Python. `geospaNN` enables geometry-aware covariance estimation and spatial prediction at scales—tens of thousands of locations—that are feasible on a personal laptop. This makes advanced spatial analysis accessible to individual researchers without specialized computing infrastructure.

The NNGP implementation within `geospaNN` also fills a notable gap in the Python ecosystem. While widely used R packages such as `spNNGP` [@finley2019efficient] and `BRISC` [@saha2018brisc] provide efficient NNGP-based spatial models, no comparable Python implementation currently exists. `geospaNN` therefore offers the first Python-based pathway for NNGP modeling in geospatial applications, meeting the growing demand for large-scale spatial analysis.

<!---
`geospaNN` is a Python package for geospatial analysis that uses NN-GLS [@zhan2024neural], a novel and scalable class of NNs explicitly designed to account for spatial correlation in the data. The package is effectively represented as a geographically-informed Graph Neural Network (GNN) by using NNGP covariance matrices. It is embedded within the PyG framework, which is designed for scalable execution of GNNs on irregular data structures like graphs. `geospaNN` is primarily intended for researchers and scientists in machine learning and geostatistics, but can also be applied more generally to estimation and prediction tasks involving other data with dependency structures, like time-series. `geospaNN` provides lightweight, user-friendly wrappers for data simulation, preprocessing, and model training, which significantly simplify the analytical pipeline. A portion of `geospaNN` has already been used in articles such as @zhan2024neural and @heaton2024adjusting. In the future, we anticipate that `geospaNN` will play a significant role at the interface of machine learning and spatial statistics, serving as a foundation for both scientific and methodological explorations.

The implementation of NNGP models within `geospaNN` is of independent importance. NNGP enables scalable covariance matrix inversions, which features extensively in geospatial models. There exist two widely used R-packages `spNNGP` [@finley2019efficient] and `BRISC` [@saha2018brisc] for implementations of NNGP, but to our knowledge there is no Python implementation of NNGP. We thus offer an avenue to efficiently analyze massive geospatial datasets in Python. --><!---using NNGP (linear models) and NN-GLS (non-linear models). based applications.-->

# State of the field
Integrating geospatial data with modern deep learning has motivated the development of several specialized Python tools. For example, `TorchGeo` [@TorchGeo2022] extends `PyTorch` [@paszke2019pytorch] for tasks such as land cover classification, object detection, and geospatial segmentation, while the R package `geodl` [@maxwell2024geodl] was recently introduced for analyzing geospatial and spatiotemporal datasets. However, these frameworks are primarily designed for raster and vector data—especially satellite imagery—rather than for general geostatistical modeling or spatial regression. Their scope is therefore limited when working with point-referenced geospatial data or when statistical interpretability is essential.

For irregular spatial data, GNNs have emerged as a powerful modeling approach. <!--GNNs operate on graph-structured inputs and learn representations through message passing, making them well-suited for spatial networks, sensor locations, or other non-gridded domains.--> `PyTorch-Geometric` (`PyG`) [@PyG2019] provides a flexible and efficient framework for implementing GNNs, and these models have been successfully applied to a range of geospatial tasks, including crop yield prediction [@fan2022gnn] and traffic flow modeling [@wang2020traffic]. Despite their popularity, there is still no unified, statistically oriented GNN software designed specifically for geospatial regression or rigorous covariance modeling. This leaves a gap between machine learning–focused GNN libraries and the needs of statistical geoscience.

GP–based tools provide another major category of spatial modeling software. `PyKrige` [@murphy2014pykrige] offers classical kriging prediction but is limited to predefined mean functions and lacks scalable covariance computation for large datasets. `GPyTorch` [@gardner2018gpytorch] supports flexible mean modeling and GP inference within a mixed-model framework, but its functionality is highly modular and requires substantial custom implementation, making it difficult for general users to apply. Moreover, its covariance approximations are not explicitly designed to exploit spatial geometry, which can reduce efficiency and accuracy compared with approaches tailored to geostatistical structure.

<!--
In Python, to integrate geospatial data with deep learning, several specialized tools have been developed. Notably, `TorchGeo` [@TorchGeo2022] extends `PyTorch` [@paszke2019pytorch] for tasks such as land cover classification, object detection, and geospatial segmentation. Independently, the R package `geodl` [@maxwell2024geodl] was recently introduced for analyzing geospatial and spatiotemporal data.  However, these tools primarily supports raster and vector data, such as satellite imagery, which limits their general applications.

GNNs are the most common approaches for data with irregular geometry. It efficiently processes graph-structured data by learning graph-level representations through message passing and aggregation. For GNNs implementation, the PyTorch Geometric (`PyG`) library provides a highly customizable framework for defining graph convolutional layers [@PyG2019]. GNNs have been widely applied in geospatial analysis, including crop yield prediction [@fan2022gnn], and traffic flow modeling [@wang2020traffic]. However, despite their growing adoption, there is still no systematic analytical GNN software tailored for broad use within the statistical community.

On the other hand, GP-based softwares such as `PyKrige`, `GPyTorch`, have been widely used for spatial modeling. However, `PyKrige` only focuses on spatial prediction. It lacks flexibility on the family of regression functions and does not provide scalable computation for large dataset. `GPyTorch` does allow simultaneous mean estimation and spatial prediction under a spatial mixed model, but in a complicated way that requires a large volume of additional self-coding, thus being hardly accessible to general users. The covariance approximation in `GPyTorch` fails to fully use the geographical information, thus being less efficient and accurate comparing to `geospaNN`.
-->

# The geospaNN Package

This section provides an overview of the `geospaNN` package, including the model architecture and several technical details. For practical examples and detailed documentation, visit the [`geospaNN` website](https://wentaozhan1998.github.io/geospaNN-doc). <!--A [vignette](https://github.com/WentaoZhan1998/geospaNN/blob/main/vignette/vignette.pdf) is also available for detailed illustration of the package.-->

## NN-GLS Overview
<!---
In a simple linear regression scenario, Gauss-Markov's Theorem states that when the data exhibits a correlation structure, generalized least
squares (GLS) provides greater efficiency than ordinary least squares (OLS). For vanilla neural networks, it is typically implicitly assumed that the
observations $Y_i$ are not correlated, and mean squared error is used as the loss function for regression tasks for continuous outcomes:
$$
Y = m(X) + \epsilon.
$$
-->
In methodology, `geospaNN`  uses NN-GLS [@zhan2024neural], a novel and scalable class of NNs explicitly designed to account for spatial correlation in the data. NN-GLS embeds NN with the following spatial mixed model:
$$
Y(s) = m(X(s)) + \epsilon(s)
$$
where $Y(s)$ and $X(s)$ are respectively the outcome and covariates observed at location $s$, $m$ is a non-linear function relating $X(s)$ to $Y(s)$ to be estimated using a NN. The key distinction from the standard non-linear regression setup is that here the errors $\epsilon(s)$ is a GP that models spatial correlation. <!---Typically it is decomposed as the sum of a Gaussian process for the spatial dependence and, possibly, a noise component. Given the observations $Y$, the covariates $X$, and the locations $s$ as the input, NN-GLS estimates the mean function $m()$ and predicts $Y$ at new locations. The ultimate goal of NN-GLS is to help understanding the complex relationship between observations and covariates and provide accurate geospatial prediction.

Let $\Sigma$ be a model for the spatial error $\epsilon$, then the well-known theory (Gauss-Markov theorem) of OLS and GLS from the statistical literature motivates the introduction of a GLS-style loss in NNs: 
$$
L\big(m(\cdot)\big) = \frac{1}{n}\big(Y - m(X)\big)^{\top}\Sigma^{-1}\big(Y - m(X)\big).
$$

However, minimizing this GLS-style loss in practice presents several challenges:

1.  The covariance matrix $\Sigma$ is based on a parametric covariance function, and the parameters are typically unknown.
2.  Even if $\Sigma$ is well-estimated, inverting $\Sigma$ becomes computationally infeasible as the sample size $n$ grows large.
3.  Since the GLS loss is not additive across observations, mini-batching---an essential technique used in implementation of modern NNs---cannot be applied directly.
-->

To solve the model, NN-GLS replaces the original loss function with a GLS-style
version, which naturally equates to a specialized GNN. For computational efficiency, NNGP is introduced to approximate the covariance,.<!--Mathematically, given a covariance matrix $\Sigma$, its inverse can be approximated as: 
$$
\Sigma^{-1} = Q = Q^{\top/2}Q^{1/2},
$$
where $Q^{1/2}$ is a lower triangular sparse matrix. The $j$-th element in the $i$-th row of $Q^{1/2}$ is non-zero if and only if location $j$ is in the $k$-nearest neighborhood of location $i$, where $k$ is a pre-specified neighborhood size. This approximation simplifies 
 On a finite sample,--> <!--NNGP sparsifies the covariance matrix, thus simplifying both likelihood computation and the GLS-style loss, addressing the computational bottlenecks within the method. Specifically, for the GLS loss function: 
$$
L\big(m(\cdot)\big) = \frac{1}{n}\big(Y - m(X)\big)^{\top}\Sigma^{-1}\big(Y - m(X)\big) = \frac{1}{n}\big(Y^* - m^*(X)\big)^{\top}\big(Y^* - m^*(X)\big) = \frac{1}{n}\sum_{i = 1}^n(Y^*_i - m^*(X_i))^2 ,
$$
where $Y^* = Q^{1/2}Y$ and $m^*(X) = Q^{1/2}m(X)$ can be obtained easily using aggregation over the nearest-neighbot directed acyclic graph specifying the NNGP approximation. The GLS loss returns to an additive form, allowing for mini-batching instead
of full-batch training. For likelihood-based parameter estimation, 
`geospaNN` uses the `BRISC` R package as an efficient solution [@saha2018brisc].--> In NN-GLS, we assume that the parameters of the covariance matrix is unknown. These covariance parameters $\theta$ for spatial process $\epsilon(s)$ and the weights parameters of the NN used to model $m$ are estimated iteratively, and training proceeds until the validation loss
converges. Once estimation is complete, nearest-neighbor-based kriging is used to generate spatial predictions at new locations. 

## Core features of `geospaNN`
The `geospaNN` workflow begins by preparing the data and constructing the model inputs. Users may supply covariates, responses, and coordinates in simple matrix form. The function `geospaNN.make_graph` then creates a `DataLoader` object that organizes the data and handles batching efficiently:
```python
data = geospaNN.make_graph(X, Y, coord, nn)
```
`geospaNN` provides flexible tools for specifying neural network architectures and defining training routines, all fully compatible with the `PyTorch` ecosystem. The code example below illustrates a typical training setup. Here, a two-layer multilayer perceptron is used to model the nonlinear mean structure. The `nngls_model` object implements the NN–GLS model, and `trainer_nngls` manages the iterative training process. Users may rely on default hyperparameters or customize them as needed.
```python
mlp_nngls = torch.nn.Sequential(
    torch.nn.Linear(p, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 1),
)
nngls_model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, 
                             mlp=mlp_nngls, theta=torch.tensor(theta0))
trainer_nngls = geospaNN.nngls_train(nngls_model, lr=0.1, min_delta=0.001)
training_log = trainer_nngls.train(data_train, data_val, epoch_num= 200, 
                                   Update_init=10, Update_step=2, 
                                   batch_size = 60, seed = 2025)
```

Once training is complete, the fitted model provides three key capabilities:

1.  estimate the non-linear mean function by $\hat{m}$.
2.  estimate the spatial parameters by $\hat{\theta}$.
3.  predict the outcome at new locations by $\hat{Y}$.

The mean function $m(x)$ represents the non-spatial component of the spatial mixed model, representing the non-spatial relationship between $Y$ and covariates $X$. To obtain predictions of the mean function for a given matrix of covariates `X`, users may call:
```python
estimate = nngls_model.estimate(X)
```

The estimated spatial parameters $\hat{\theta}$ characterize the spatial correlation structure implied by the model. They are stored internally and can be accessed directly:
```python
nngls_model.theta
```
These parameters can be used to reconstruct the implied covariance matrix or inform further geostatistical analyses.

While mean function estimation reflect the connection between variables, to predict the value of response $Y$ at new locations with uncertainty quantification, `geospaNN` uses `predict()` method:
```python
[test_predict, test_PI_U, test_PI_L] = nngls_model.predict(data_train, data_test, 
                                                           PI = True)
```

## Other Features

In addition to estimation and prediction for the NN-GLS spatial mixed models, `geospaNN` offers a suite of additional features that support a wide range of geospatial analyses. <!--Given an $n \times n$ covariance matrix $\Sigma$ and a $k$-neighbor list (defaulting to nearest
neighbors), our implementation guarantees $O(n)$ computational
complexity for any approximate matrix products involving $\Sigma^{1/2}$,
$\Sigma^{-1/2}$, and $\Sigma^{-1}$. The option of customizing the $k$-neighbor list is provided in case specially designed neighbor set is desired, like using distant neighbors to model long-range correlation [@stein2004approximating], or neighbors based on geodesic-distances if working with data collected on a non-Euclidean domain like water bodies [@gilbert2024visibility]. NNGP is fundamental to several key
scalable features in `geospaNN`, including spatial data simulation and
kriging.--> `geospaNN` provides simulation module allowing users to customize the spatial parameters and mean functions to generate $Y$, $X$, and $s$. Users are allowed to customize the spatial coordinates to simulate under different context. <!--This feature can be flexibly used to simulate a Gaussian process as a general spatial random effect by specifying a zero mean function. For kriging, instead of using the full observed set, only the nearest neighbors of each new spot in the observed set are used to compute the kriging weights.--> `geospaNN` implements nearest neighbor kriging, an alternate to full kriging, which has been shown in @zhan2024neural to guarantee accurate prediction interval under various settings. For essential machine learning tasks, `geospaNN` offers modules including NN architecture design, training log report, and result visualization. <!--The training log visualizes the evolution of validation loss as well as spatial parameters, providing deeper insight into model convergence. To interpret high-dimensional result, `geospaNN` incorporates partial dependence plots (PDPs) to illustrate the marginal effect of individual covariates on the outcome. When compared across multiple models, PDPs offer invaluable insights into the complex relationships between covariates and the predicted outcome.--><!--As a special case of the spatial non-linear mixed model, SPLMM with NNGP covariance is also implemented in `geospaNN`. The implementation here is--> `geospaNN` also implements spatial linear mixed model (SPLMM) as a special case of NN-GLS. It should be an optimal choice for the Python users if efficient SPLMM solution is wanted for large geospatial datasets.

Because the above code snippets rely on additional setup, they are not meant to run independently. Users can find complete, reproducible examples and detailed documentation in the project [vignette](https://github.com/WentaoZhan1998/geospaNN/blob/main/vignette/vignette.pdf).

# Discussion

The `geospaNN` package provides a machine learning toolkit for geostatistical analysis. Built on an efficient implementation of the NN–GLS approach proposed in @zhan2024neural, `geospaNN` supports a range of core statistical tasks, including nonlinear mean-function estimation, covariance parameter estimation, and spatial prediction with uncertainty quantification. Leveraging the sparsity of the NNGP approximation, the software integrates naturally into the GNN framework, enabling the use of graph-based operations and opening the door to more advanced neural architectures in geospatial modeling.

Despite these strengths, the current version of `geospaNN` has several limitations. At present, the package supports only a limited set of stationary, parametric covariance models and does not handle non-stationary or non-Gaussian spatial processes. The neural network component is designed and tested mainly for simple architectures such as multilayer perceptrons, which work well for moderate-scale spatial data but limit applicability to more complex or high-dimensional input structures. One important future direction is to increase the flexibility of our model and add features to the main steps in `geospaNN` to adopt more general estimation and prediction tasks. In addition, `geospaNN` currently requires R-dependency and does not support GPU acceleration. In the future releases, we will address these key issues to further improve the performance of the software. 

Conceptually, a longer-term direction for `geospaNN` is to evolve into a general framework for geospatially informed deep learning, where spatially structured message passing can be incorporated while maintaining statistical interpretability. We also plan to extend the methodology to additional data types and distributional settings beyond the current Gaussian framework.

# Acknowledgements

This work is supported by National Institute of Environmental Health
Sciences grant R01ES033739. The authors report there are no competing
interests to declare.

# References
