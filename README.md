# GeoNets
GeoNets package repository for the [paper](https://arxiv.org/pdf/2304.09157.pdf) "neural networks for geospatial data".
=======
This is the package repository for the method proposed in the NN-GLS paper. To install (currently), use the following command:

```commandline\
pip install git+https://github.com/WentaoZhan1998/NN-GLS.git#egg=geoNets
```

## An easy pipeline:

First import the modules and set up the parameters
```commandline\
import torch
import geospaNN
import numpy as np

# This line defines the Friedman's function.
def f5(X): return (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] +5*X[:,4])/6

sigma = 1
phi = 3
tau = 0.01
theta = torch.tensor([sigma, phi / np.sqrt(2), tau])

p = 5; funXY = f5

n = 1000
nn = 20
batch_size = 50
```

Next, simulate and split the data.
```commandline\
torch.manual_seed(2024)
X, Y, coord, cov, corerr = geospaNN.Simulation(n, p, nn, funXY, theta, range=[0, 10])
data = geospaNN.make_graph(X, Y, coord, nn)
data_train, data_val, data_test = geospaNN.split_data(X, Y, coord, neighbor_size=20,
                                                   test_proportion=0.2)
```    

Compose the mlp structure and train easily.
```commandline\                                      
mlp = torch.nn.Sequential(
    torch.nn.Linear(p, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
)
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp, theta=torch.tensor([1.5, 5, 0.1]))
nngls_model = geospaNN.nngls_train(model, lr =  0.01, min_delta = 0.001)
training_log = nngls_model.train(data_train, data_val, data_test,
                                 Update_init = 10, Update_step = 10)
```

Estimation from the model.
```commandline\
train_estimate = model.estimate(data_train.x)
```

Kriging prediction from the model.
```commandline\
test_predict = model.predict(data_train, data_test)
```


## Running examples:
* A simulation experiment with a common spatial setting is shown [here](https://github.com/WentaoZhan1998/NN-GLS/blob/main/Example_simulation.ipynb)

* A real data experiment is shown [here](https://github.com/WentaoZhan1998/NN-GLS/blob/main/Example_realdata.ipynb). 
    * The PM2.5 data is collected from the [U.S. Environmental Protection Agency](https://www.epa.gov/outdoor-air-quality-data/download-daily-data) datasets for each state are collected and bound together to obtain 'pm25_2022.csv'. daily PM2.5 files are subsets of 'pm25_2022.csv' produced by 'realdata_preprocess.py'. One can skip the preprocessing and use the daily files directory. 
    * The meteorological data is collected from the [National Centers for Environmental Prediction’s (NCEP) North American Regional Reanalysis (NARR) product](https://psl.noaa.gov/data/gridded/data.narr.html). The '.nc' (netCDF) files should be downloaded from the website and saved in the root directory to run 'realdata_preprocess.py'. Otherwise, one may skip the preprocessing and use covariate files directly. 


