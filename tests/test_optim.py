import scipy

import torch
import geospaNN
import numpy as np
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

def import_BRISC():
    BRISC = importr('BRISC')
    return BRISC
BRISC = import_BRISC()
def BRISC_estimation(residual, X, coord, search_type = "brute"):
    residual_r = robjects.FloatVector(residual)
    coord_r = robjects.FloatVector(coord.transpose().reshape(-1))
    coord_r = robjects.r['matrix'](coord_r, ncol=2)

    if X is None:
        res = BRISC.BRISC_estimation(coords = coord_r, y = residual_r, search_type = search_type)
    else:
        Xr = robjects.FloatVector(X.transpose().reshape(-1))
        Xr = robjects.r['matrix'](Xr, ncol=X.shape[1])
        res = BRISC.BRISC_estimation(coords = coord_r, y = residual_r, x = Xr, search_type = search_type)

    theta_hat = res[9]
    beta = res[8]
    beta = np.array(beta)
    theta_hat = np.array(theta_hat)
    phi = theta_hat[2]
    tau_sq = theta_hat[1]
    sigma_sq = theta_hat[0]
    theta_hat[1] = phi
    theta_hat[2] = tau_sq / sigma_sq

    return beta, theta_hat

def f5(X): return (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] +5*X[:,4])/6
def f1(X): return 5*X + 2

sigma = 1
phi = 3
tau = 0.01
theta = torch.tensor([sigma, phi / np.sqrt(2), tau])

p = 1; funXY = f1

n = 1000
nn = 20
batch_size = 50

torch.manual_seed(2024)
X, Y, coord, cov, corerr = geospaNN.Simulation(n, p, nn, funXY, theta, range=[0, 10])

data = geospaNN.make_graph(X, Y, coord, nn)

torch.manual_seed(2024)
np.random.seed(0)
data_train, data_val, data_test = geospaNN.split_data(X, Y, coord, neighbor_size = nn,
                                                   test_proportion = 0.2)

torch.manual_seed(2024)
mlp_nn = torch.nn.Sequential(
    torch.nn.Linear(p, 1)
)
nn_model = geospaNN.nn_train(mlp_nn, lr =  0.01, min_delta = 0.001)
training_log = nn_model.train(data_train, data_val, data_test)
theta0 = geospaNN.theta_update(torch.tensor([1, 1.5, 0.01]), mlp_nn(data_train.x).squeeze() - data_train.y, data_train.pos, neighbor_size = 20)
mlp_nngls = torch.nn.Sequential(
    torch.nn.Linear(p, 1),
)
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp_nngls, theta=torch.tensor(theta0))
nngls_model = geospaNN.nngls_train(model, lr =  0.01, min_delta = 0.001)
training_log = nngls_model.train(data_train, data_val, data_test,
                                 Update_init = 10, Update_step = 5)

coord = data_train.pos.detach().numpy()
w = data_train.y - model.estimate(data_train.x)
neighbor_size = 20

n_train = w.shape[0]
dist = geospaNN.distance_np(coord, coord)
rank = geospaNN.make_rank(coord, neighbor_size)
print('Theta updated from')
print(theta0)

def likelihood(theta):
    sigma, phi, tau = theta
    cov = sigma * np.exp(-phi * dist) + tau * sigma * np.eye(n_train)  # Precompute cov

    term1 = 0
    term2 = 0
    eye_n_train = np.eye(n_train)  # Precompute identity matrix once

    for i in range(n_train):
        ind = rank[i, :][rank[i, :] <= i]
        id = np.append(ind, i)

        sub_cov = cov[np.ix_(ind, ind)]  # Efficient sub-matrix indexing

        # Use Cholesky solve if positive definite
        try:
            bi = np.linalg.solve(sub_cov, cov[ind, i])
            #bi = scipy.linalg.cho_solve(scipy.linalg.cho_factor(sub_cov), cov[ind, i])
        except scipy.linalg._flapack.__flapack_error or np.linalg.LinAlgError:
            bi = np.zeros(ind.shape)

        I_B_i = np.append(-bi, 1)
        F_i = cov[i, i] - np.inner(cov[ind, i], bi)
        decor_res = np.sqrt(1 / F_i) * np.dot(I_B_i, w[id])
        term1 += np.log(F_i)
        term2 += decor_res ** 2

    return term1 + term2

start_time = time.time()
res = scipy.optimize.minimize(likelihood, theta0, method='L-BFGS-B',
                              bounds=[(0, None), (0, None), (0, None)])
print(res.x)
print(str(time.time() - start_time) + " seconds")

start_time = time.time()
res = BRISC_estimation(w, None, coord, search_type="tree")
print(res)
print(str(time.time() - start_time) + " seconds")