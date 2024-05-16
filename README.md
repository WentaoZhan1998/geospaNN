[![PyPI](https://img.shields.io/pypi/v/geospaNN?logo=PyPI)](https://pypi.org/project/geospaNN)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/geospaNN)

# GeospaNN - Neural networks for geospatial data
**Authors**: Wentao Zhan (<wzhan3@jhu.edu>), Abhirup Datta (<abhidatta@jhu.edu>)
## A package based on the paper: [Neural networks for geospatial data](https://arxiv.org/pdf/2304.09157.pdf)
**GeospaNN** is a formal implementation of NN-GLS, the Neural Networks for geospatial data proposed in Zhan et.al (2023), 
that explicitly accounts for spatial correlation in the data. The package is developed using [PyTorch](https://pytorch.org/) and 
under the framework of [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) library. 
NN-GLS is a geographically-informed Graph Neural Network (GNN) for analyzing large and irregular geospatial data, 
that combines multi-layer perceptrons, Gaussian processes, and generalized least squares (GLS) loss. 
NN-GLS offers both regression function estimation and spatial prediction, and can scale up to sample sizes of hundreds of thousands. 
Users are welcome to provide any helpful suggestions and comments.

## Overview
The Python package **geospaNN** stands for 'geospatial Neural Networks', where we implement NN-GLS, 
neural networks tailored for analysis of geospatial data that explicitly accounts for spatial dependence (Zhan et.al, 2023). 
Geospatial data naturally exhibits spatial correlation or dependence and traditional geostatistical analysis often relies on 
model-based approaches to handle the spatial dependency, treating the spatial outcome $y(s)$ as a linear regression on covariates $x(s)$ and 
modeling dependency through the spatially correlated errors. 
For example, using Gaussian processes (GP) to model dependent errors, 
simple techniques like kriging can provide powerful prediction performance by properly aggregating the neighboring information. 
On the other hand, artificial Neural Networks (NN), one of the most popular machine learning approaches, could be used to estimate non-linear regression functions. 
However, common neural networks like multi-layer perceptrons (MLP) does not incorporate correlation among data units.

Our package **geospaNN** takes the advantages from both perspectives and provides an efficient tool for geospatial data analysis. 
In NN-GLS, an MLP is used to model the non-linear regression function while a GP is used to model the spatial dependence. 
The resulting loss function then becomes a generalized least squares (GLS) loss informed by the GP covariance matrix, 
thereby explicitly incorporating spatial correlation into the neural network optimization. 
The idea mimics the extension of ordinary least squares (OLS) loss to GLS loss in linear regression for dependent data.

Zhan and Datta, 2023 shows that neural networks with GLS loss can be represented as a graph neural network, 
with the GP covariances guiding the neighborhood aggregation on the output layer. 
Thus NN-GLS is implemented in **geospaNN** with the framework of Graph Neural Networks (GNN), and is highly generalizable. 
(The implementation of geospaNN' uses the 'torch_geom' module.)

**geospaNN** provides an estimate of regression function 𝑓(𝑥) as well as accurate spatial predictions using Gaussian process (kriging), 
and thus constitutes a complete geospatial analysis pipeline. 
To accelerate the training process for the GP, **geospaNN** approximates the working correlation structure using 
Nearest Neighbor Gaussian Process (NNGP) (Datta et al., 2016) which makes it suitable for larger datasets towards a size of 1 million.

<div align="center">
<a href="https://arxiv.org/pdf/2304.09157.pdf">
  <img
    src="https://github.com/WentaoZhan1998/geospaNN/blob/main/data/nngls.png?raw=True"
    width="800"
  >
</a>
</div>

## Installation
(Currently) to install the development version of the package, a pre-installed PyTorch and PyG libraries are needed. Installation in the following order is recommended to avoid any compilation issue.
1. To install PyTorch, find and install the binary suitable for your machine [here](https://pytorch.org/).
2. Then to install the PyG library, find and install the proper binary [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
3. Make sure to also install the dependencies including *pyg_lib*, *torch_scatter*, *torch_sparse*, *torch_cluster*, and *torch_spline_conv*.

Once PyTorch and PyG are successfully installed, use the following command in the terminal for the latest version:
```commandline\
pip install https://github.com/WentaoZhan1998/geospaNN/archive/main.zip
```

To install the pypi version, use the following command in the terminal:
```commandline\
pip install geospaNN
```

## An easy running sample:

First import the modules and set up the parameters
1. Define the Friedman's function, and specify the dimension of input covariates.
2. Set the parameters for the spatial process.
3. Set the hyperparameters of the data.
```commandline\
import torch
import geospaNN
import numpy as np

# 1.
def f5(X): return (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] +5*X[:,4])/6

p = 5; funXY = f5

# 2.
sigma = 1
phi = 3/np.sqrt(2)
tau = 0.01
theta = torch.tensor([sigma, phi, tau])

# 3.
n = 1000            # Size of the simulated sample.
nn = 20             # Neighbor size used for NNGP.
batch_size = 50     # Batch size for training the neural networks.
```

Next, simulate and split the data.
1. Simulate the spatially correlated data with spatial coordinates randomly sampled on a [0, 10]^2 squared domain.
2. Build the nearest neighbor graph, as a torch_geometric.data.Data object.
3. Split data into training, validation, testing sets.
```commandline\
# 1.
torch.manual_seed(2024)
X, Y, coord, cov, corerr = geospaNN.Simulation(n, p, nn, funXY, theta, range=[0, 10])

# 2.
data = geospaNN.make_graph(X, Y, coord, nn)

# 3.
data_train, data_val, data_test = geospaNN.split_data(X, Y, coord, neighbor_size=20,
                                                   test_proportion=0.2)
```    

Compose the mlp structure and train easily.
1. Define the mlp structure (torch.nn) to use.
2. Define the NN-GLS corresponding model.
3. Define the NN-GLS training class with learning rate and tolerance.
4. Train the model.
```commandline\
# 1.             
mlp = torch.nn.Sequential(
    torch.nn.Linear(p, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
)

# 2.
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp, theta=torch.tensor([1.5, 5, 0.1]))

# 3.
nngls_model = geospaNN.nngls_train(model, lr =  0.01, min_delta = 0.001)

# 4.
training_log = nngls_model.train(data_train, data_val, data_test,
                                 Update_init = 10, Update_step = 10)
```

Estimation from the model. The variable is a torch.Tensor object of the same dimension
```commandline\
train_estimate = model.estimate(data_train.x)
```

Kriging prediction from the model. The first variable is supposed to be the data used for training, and the second 
variable a torch_geometric.data.Data object which can be composed by geospaNN.make_graph()'.
```commandline\
test_predict = model.predict(data_train, data_test)
```

## Running examples:
* A simulation experiment with a common spatial setting is shown [here](https://github.com/WentaoZhan1998/geospaNN/blob/main/Example_simulation.ipynb).

* A real data experiment is shown [here](https://github.com/WentaoZhan1998/geospaNN/blob/main/Example_realdata.ipynb). 
* The PM2.5 data is collected from the [U.S. Environmental Protection Agency](https://www.epa.gov/outdoor-air-quality-data/download-daily-data) datasets for each state are collected and bound together to obtain 'pm25_2022.csv'. daily PM2.5 files are subsets of 'pm25_2022.csv' produced by 'realdata_preprocess.py'. One can skip the preprocessing and use the daily files directory. 
* The meteorological data is collected from the [National Centers for Environmental Prediction’s (NCEP) North American Regional Reanalysis (NARR) product](https://psl.noaa.gov/data/gridded/data.narr.html). The '.nc' (netCDF) files should be downloaded from the website and saved in the root directory to run 'realdata_preprocess.py'. Otherwise, one may skip the preprocessing and use covariate files directly. 

## Citation
Please cite the following paper when you use **geospaNN**:

> Zhan, Wentao, and Abhirup Datta. "Neural networks for geospatial data." Journal of the American Statistical Association Theory and Methods (2024, in press) arXiv preprint arXiv:2304.09157
 

## References

Datta, Abhirup, Sudipto Banerjee, Andrew O. Finley, and Alan E. Gelfand. "Hierarchical nearest-neighbor Gaussian process models for large geostatistical datasets." Journal of the American Statistical Association 111, no. 514 (2016): 800-812. [link](https://www.tandfonline.com/doi/full/10.1080/01621459.2015.1044091)

Zhan, Wentao, and Abhirup Datta. "Neural networks for geospatial data." Journal of the American Statistical Association Theory and Methods (2024, in press) arXiv preprint arXiv:2304.09157 [link](https://arxiv.org/abs/2304.09157)
