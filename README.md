# NN-GLS
NNGLS package repository for the [NN-GLS paper](https://arxiv.org/pdf/2304.09157.pdf)
=======
This is the package repository for the method proposed in the NN-GLS paper. To install (currently), use the following command:

```commandline\
pip install git+https://github.com/WentaoZhan1998/NN-GLS.git#egg=nngls
```

## An easy pipeline:

First import the modules and set up the parameters
```commandline\
import torch
import nngls
import numpy as np

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
X, Y, coord, cov, corerr = nngls.Simulation(n, p, nn, funXY, theta, range=[0, 10])
data = nngls.make_graph(X, Y, coord, nn)
data_train, data_val, data_test = nngls.split_data(X, Y, coord, neighbor_size=20,
                                                   test_proportion=0.2)
```    

Finally, compose the mlp structure and train easily.
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
model = nngls.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp, theta=torch.tensor([1.5, 5, 0.1]))
nngls_model = nngls.nngls_train(model, lr =  0.01, min_delta = 0.001)
training_log = nngls_model.train(data_train, data_val, data_test,
                                 Update_init = 10, Update_step = 10)
```

