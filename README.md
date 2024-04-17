# GeospaNN
## Package based on the paper: [Neural networks for geospatial data](https://arxiv.org/pdf/2304.09157.pdf)

This is the package repository for the method proposed in the paper. To install the github version, use the following command:
```commandline\
pip install git+https://github.com/WentaoZhan1998/geospaNN.git#egg=geospaNN
```

To install the pip version, use the following command:
```commandline\
pip install geospaNN
```

## An easy pipeline for a simulation experiment:

First import the modules and set up the parameters
```commandline\
import torch
import geospaNN
import numpy as np

# Define the Friedman's function, and specify the dimension of input covariates.
def f5(X): return (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] +5*X[:,4])/6
p = 5; funXY = f5

# Set the parameters for the spatial process.
sigma = 1
phi = 3/np.sqrt(2)
tau = 0.01
theta = torch.tensor([sigma, phi, tau])

n = 1000 # Size of the simulated sample.
nn = 20 # Neighbor size used for NNGP.
batch_size = 50 # Batch size for training the neural networks.
```

Next, simulate and split the data.
```commandline\
torch.manual_seed(2024)
# Simulate the spatially correlated data with spatial coordinates randomly sampled on a [0, 10]^2 squared domain.
X, Y, coord, cov, corerr = geospaNN.Simulation(n, p, nn, funXY, theta, range=[0, 10])

# Build the nearest neighbor graph, as a torch_geometric.data.Data object.
data = geospaNN.make_graph(X, Y, coord, nn)

# Split data into training, validation, testing sets.
data_train, data_val, data_test = geospaNN.split_data(X, Y, coord, neighbor_size=20,
                                                   test_proportion=0.2)
```    

Compose the mlp structure and train easily.
```commandline\
# Define the mlp structure (torch.nn) to use.                         
mlp = torch.nn.Sequential(
    torch.nn.Linear(p, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
)

# Define the NN-GLS corresponding model. 
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp, theta=torch.tensor([1.5, 5, 0.1]))

# Define the NN-GLS training class with learning rate and tolerance.
nngls_model = geospaNN.nngls_train(model, lr =  0.01, min_delta = 0.001)

# Train the model.
training_log = nngls_model.train(data_train, data_val, data_test,
                                 Update_init = 10, Update_step = 10)
```

Estimation from the model. The variable is a torch.Tensor object of the same dimension
```commandline\
train_estimate = model.estimate(data_train.x)
```

Kriging prediction from the model. The first variable is supposed to be the data used for training, and the second 
variable a torch_geometric.data.Data object which can be composed by 'geospaNN.make_graph()'.
```commandline\
test_predict = model.predict(data_train, data_test)
```

## Running examples:
* A simulation experiment with a common spatial setting is shown [here](https://github.com/WentaoZhan1998/NN-GLS/blob/main/Example_simulation.ipynb)

* A real data experiment is shown [here](https://github.com/WentaoZhan1998/NN-GLS/blob/main/Example_realdata.ipynb). 
* The PM2.5 data is collected from the [U.S. Environmental Protection Agency](https://www.epa.gov/outdoor-air-quality-data/download-daily-data) datasets for each state are collected and bound together to obtain 'pm25_2022.csv'. daily PM2.5 files are subsets of 'pm25_2022.csv' produced by 'realdata_preprocess.py'. One can skip the preprocessing and use the daily files directory. 
* The meteorological data is collected from the [National Centers for Environmental Prediction’s (NCEP) North American Regional Reanalysis (NARR) product](https://psl.noaa.gov/data/gridded/data.narr.html). The '.nc' (netCDF) files should be downloaded from the website and saved in the root directory to run 'realdata_preprocess.py'. Otherwise, one may skip the preprocessing and use covariate files directly. 
