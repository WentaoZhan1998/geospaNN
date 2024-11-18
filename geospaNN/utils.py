from __future__ import annotations
from .R import BRISC_estimation

import numpy as np
import torch
import scipy
from sklearn.neighbors import NearestNeighbors
from typing import Callable, Optional, Tuple
from sklearn.model_selection import train_test_split
import torch_geometric
from torch_geometric.loader import NeighborLoader
import math
import warnings
import random

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.

    Attributes:
        optimizer:
            the optimizer we are using
        patience (int):
            How many epochs to wait before updating the lr. Default being 5.
        min_lr (float):
            Least lr value to reduce to while updating.
        factor (float):
            Factor by which the lr should be updated, i.e. new_lr = old_lr * factor.
    """
    def __init__(
        self,
            optimizer,
            patience: Optional[int] = 5,
            min_lr: Optional[float] = 1e-6,
            factor: Optional[float] = 0.5
    ):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

class DropoutLayer(torch.nn.Module):
    """
    Customized dropout layer where the nodes values are dropped with probability p.

    Attributes:
        p (float):
            The drop probability
    """
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            u1 = (np.random.rand(*input.shape)<self.p) / self.p
            u1 *= u1
            return u1
        else:
            input *= self.p

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.

    Attributes:
        patience (int):
            How many epochs to wait before stopping when loss is not improving
        min_delta (float):
            Minimum difference between new loss and old loss for new loss to be considered as an improvement
    """
    def __init__(self,
                 patience: Optional[int] = 5,
                 min_delta: Optional[float] = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta or math.isnan(self.best_loss):
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            #print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class Sparse_B():
    """
    A sparse representation of the lower-triangular neighboring nxn matrix B_dense,
    where each row contains at most p non-zero values.

    Attributes:
        B :
            nxp array contains all non-zero values in B_dense.
        n : 
            The number of samples.
        neighbor_size :
            i.e. k in the documentation, the largest number of non-zero values in each row of B_dense.
        Ind_list :
            The nxp index array indicating the location where values in B was in B_dense. For example, the [i,j]'s index is k
            means that B_dense[i,k] = B[i,j].

    Methods:
        to_numpy():
            Transform B to np.array

        to_tensor():
            Transform B to torch.Tensor

        matmul(X, idx = None):
            Calculate the matrix product of B_dense[idx,:] and X

        invmul(y):
            Calculate the matrix-vector product of B_dense^{-1} and y. Only used when the diagonal of B_dense is constantly 1.

        Fmul(F_diag):
            Return a new Sparse_B object where B_dense is replaced by the matrix product of F * B_dense.
            F_diag is the vector representation of the nxn diagonal matrix F.

        to_dense():
            Return the dense form of B_dense as an np.array object.
    """
    def __init__(self,
                 B: torch.Tensor | np.array,
                 Ind_list: np.array):
        self.B = B
        self.n = B.shape[0]
        self.neighbor_size = B.shape[1]
        self.Ind_list = Ind_list.astype(int)

    def to_numpy(self):
        if torch.is_tensor(self.B):
           self.B = self.B.detach().numpy()
        return self

    def to_tensor(self):
        if isinstance(self.B, np.ndarray):
            self.B = torch.from_numpy(self.B).float()
        return self

    def matmul(self, X, idx = None):
        if idx == None: idx = np.array(range(self.n))
        if torch.is_tensor(X):
            self.to_tensor()
            result = torch.empty((len(idx)))
            for k in range(len(idx)):
                i = idx[k]
                ind = self.Ind_list[i,:][self.Ind_list[i,:] >= 0]
                result[k] = torch.dot(self.B[i,range(len(ind))].squeeze(),X[ind])
        elif isinstance(X, np.ndarray):
            self.to_numpy()
            if np.ndim(X) == 1:
                result = np.empty((len(idx)))
                for k in range(len(idx)):
                    i = idx[k]
                    ind = self.Ind_list[i, :][self.Ind_list[i, :] >= 0]
                    result[k] = np.dot(self.B[i, range(len(ind))].reshape(-1), X[ind])
            elif np.ndim(X) == 2:
                result = np.empty((len(idx), X.shape[1]))
                for k in range(len(idx)):
                    i = idx[k]
                    ind = self.Ind_list[i, :][self.Ind_list[i, :] >= 0]
                    #result[i,:] = np.dot(self.B[i, range(len(ind))].reshape(-1), C_Ni[ind, :])
                    result[k,:] = np.dot(self.B[i, range(len(ind))].reshape(-1), X[ind,:])
        return result

    def invmul(self, y):
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        y = y.reshape(-1)
        assert self.n == y.shape[0]
        x = y[0].unsqueeze(0)
        assert (self.B[:,0] == torch.ones(self.n)).all(), 'Only applies to I-B matrix'
        Indlist = self.Ind_list[:, 1:]
        B = -self.B[:, 1:]
        for i in range(1, self.n):
            ind = Indlist[i, :]
            id = ind >= 0
            if sum(id) == 0:
                x = torch.cat((x, y[i].unsqueeze(0)), dim = -1)
            else:
                x = torch.cat((x, (y[i] + torch.dot(x[ind[id]], B[i, :][id])).unsqueeze(0)), dim = -1)
        return x

    def Fmul(self, F_diag):
        res = Sparse_B(self.B.copy(), self.Ind_list.copy())
        for i in range(self.n):
            res.B[i,:] = F_diag[i]*self.B[i,:]
        return res

    def to_dense(self):
        B = np.zeros((self.n, self.n))
        for i in range(self.n):
            ind = self.Ind_list[i, :][self.Ind_list[i, :] >= 0]
            if len(ind) == 0:
                continue
            B[i, ind] = self.B[i,range(len(ind))]
        return B

class NNGP_cov(Sparse_B):
    """
    A subclass of Sparse_B designed for the decorrelation using NNGP. The whole object is an NNGP approximation of the
    inverse square-root of a covariance matrix \Sigma. i.d. F^{-1/2}(I-B) ~ \Sigma^{-1/2}
    ...

    Attributes:
        F_diag : torch.Tensor
            A vector of length n representing the diagonal matrix F.

    Methods:
        correlate(x):
            Approximately correlate the matrix X to \Sigma^{1/2}X.

        decorrelate(x):
            Approximately decorrelate the matrix X to \Sigma^{-1/2}X.
    """
    def __init__(self, B, F_diag, Ind_list):
        super().__init__(B = B, Ind_list = Ind_list)
        assert len(F_diag) == B.shape[0]
        self.F_diag = F_diag

    def correlate(self, x: torch.Tensor
                  ) -> torch.Tensor:
        """
        Approximately correlate the matrix X to \Sigma^{1/2}X by calculating (I_B)^{-1}F^{1/2}X.

        Parameters:
            x : Input tensor X

        Returns:
            x_cor: Correlated X
        """
        assert x.shape[0] == self.n
        return self.invmul(torch.sqrt(self.F_diag) * x)

    def decorrelate(self, x: torch.Tensor
                    ) -> torch.Tensor:
        """
        Approximately decorrelate the matrix X to \Sigma^{-1/2}X by calculating F^{-1/2}(I_B)X.

        Parameters:
            x : Input tensor X

        Returns:
            x_decor: Decorrelated X
        """
        assert x.shape[0] == self.n
        return torch.sqrt(torch.reciprocal(self.F_diag))*self.matmul(x)

"""
class NNGP_cov_np(Sparse_B):
    def __init__(self, B, F_diag, Ind_list):
        super().__init__(B = B, Ind_list = Ind_list)
        assert len(F_diag) == B.shape[0]
        self.F_diag = F_diag

    def decorrelate(self, x):
        assert x.shape[0] == self.n
        return np.sqrt(np.reciprocal(self.F_diag))*self.matmul(x)
"""

def rmvn(mu: torch.Tensor,
         cov: torch.Tensor | NNGP_cov,
         sparse: Optional[bool] = True) \
        -> torch.Tensor:
    """Randomly generate sample from multivariate normal distribution

    Generate random sample from a multivariate normal distribution with specified mean and covariance.

    Parameters:
        mu:
            The additional mean of multivariate normal of length n.
        cov:
        `   The nxn covariance matrix. When use torch.Tensor for the dense representation, Cholesky's decomposition is used
            for correlating the i.i.d normal sample. When use NNGP_cov object for sparse representation, use NNGP to approximate
            the correlating process. Dense representation is not recommended for large sample size.
        sparse:
            Designed for sparse representation, not implemented yet.

    Returns:
        sample:
            A random sample from the multivariate normal distribution.
    """
    n = len(mu) #### Check dimensionality
    if isinstance(cov, torch.Tensor):
        if n >= 2000: warnings.warn("Too large for cholesky decomposition, please try to use NNGP")
        D = torch.linalg.cholesky(cov)
        res = torch.matmul(torch.randn(1, n), D.t()) + mu
    elif isinstance(cov, NNGP_cov):
        if sparse:
            res = cov.correlate(torch.randn(1, n).reshape(-1)) + mu
        else:
            warnings.warn("To be implemented.")
    else:
        warnings.warn("Covariance matrix should be in the format of torch.Tensor or NNGP_cov.")
        return

    return  res.reshape(-1)

def make_rank(coord: torch.Tensor,
              neighbor_size: int,
              coord_ref: Optional = None
              ) -> np.ndarray:
    """Compose the nearest neighbor index list based on the coordinates.

    Find the indexes of nearest neighbors in reference set for each location i in the main set.
    The index is based on the increasing order of the distances between ith location and the locations in the reference set.

    Parameters:
        coord:
            The nxd coordinates array of target locations.
        neighbor_size:
        `   Suppose neighbor_size = k, only the top k-nearest indexes will be returned.
        coord_ref:
            The n_refxd coordinates array of reference locations. If None, use the target set itself as the reference.
            (Any location's neighbor does not include itself.)

    Returns:
        rank_list:
            A nxp array. The ith row is the indexes of the nearest neighbors for the ith location, ordered by the distance.
    """
    if coord_ref is None: neighbor_size += 1
    knn = NearestNeighbors(n_neighbors=neighbor_size)
    knn.fit(coord)
    if coord_ref is None:
        coord_ref = coord
        rank = knn.kneighbors(coord_ref)[1]
        return rank[:, 1:]
    else:
        rank = knn.kneighbors(coord_ref)[1]
        return rank[:, 0:]

def distance(coord1: torch.Tensor,
             coord2: torch.Tensor
             ) -> torch.Tensor:
    """Distance matrix between two sets of points

    Calculate the pairwise distance between two sets of locations.

    Parameters:
        coord1:
            The nxd coordinates array for the first set.
        coord12:
            The nxd coordinates array for the second set.

    Returns:
        dist:
            The distance matrix.
    """
    if coord1.ndim == 1:
        m = 1
        coord1 = coord1.unsqueeze(0)
    else:
        m = coord1.shape[0]
    if coord2.ndim == 1:
        n = 1
        coord2 = coord2.unsqueeze(0)
    else:
        n = coord2.shape[0]

    #### Can improve (resolved)
    coord1 = coord1.unsqueeze(0)
    coord2 = coord2.unsqueeze(1)
    dists = torch.sqrt(torch.sum((coord1 - coord2) ** 2, axis=-1))
    return dists

def distance_np(coord1: np.array,
                coord2: np.array
                ) -> np.array:
    """The numpy version of distance()

    Calculate the pairwise distance between two sets of locations.

    Parameters:
        coord1:
            The nxd coordinates array for the first set.
        coord2:
            The nxd coordinates array for the second set.

    Returns:
        dists:
            The distance matrix.

    See Also:
        distance : Distance matrix between two sets of points
    """
    m = coord1.shape[0]
    n = coord2.shape[0]
    #### Can improve (resolved)
    coord1 = coord1
    coord2 = coord2
    dists = np.zeros((m, n))
    for i in range(m):
        dists[i, :] = np.sqrt(np.sum((coord1[i] - coord2) ** 2, axis=1))
    return dists

def make_bf_from_cov(cov: torch.Tensor,
                     neighbor_size: int,
                     ) -> Sparse_B:
    """Obtain NNGP approximation of a covariance matrix

    Find the upper triangular matrix B and diagonal matrix F such that (I-B)'F^{-1}(I-B) appriximate the precision matrix
    (inverse of the covariance matrix). The level of approximation increase with the neighbor size. When using the full neighbor,
    the NNGP appriximation degrade to the Cholesky decomposition. (see https://arxiv.org/abs/2102.13299 for more details.)

    Parameters:
        cov:
            The nxn covariance matrix.
        neighbor_size:
            The number of nearest neighbors used for NNGP approximation.

    Returns:
        I_B: Sparse_B
            The sparse representation of I-B.
        F: torch.Tensor
            The vector representation of the diagonal matrix.

    See Also:
        make_bf : Obtain NNGP approximation with implicit covariance matrix \

        Datta, Abhirup, et al. "Hierarchical nearest-neighbor Gaussian process models for large geostatistical datasets."
        Journal of the American Statistical Association 111.514 (2016): 800-812.

        Datta, Abhirup. "Sparse nearest neighbor Cholesky matrices in spatial statistics."
        arXiv preprint arXiv:2102.13299 (2021).
    """
    n = cov.shape[0]
    B = torch.zeros((n, neighbor_size))
    ind_list = np.zeros((n, neighbor_size)).astype(int) - 1
    F = torch.zeros(n)
    rank = np.argsort(-cov, axis=-1) #### consider replace using make_rank?
    rank = rank[:, 1:(neighbor_size + 1)]
    for i in range(n):
        F[i] = cov.diag()[i]
        ind = rank[i, :][rank[i, :] <= i]
        if len(ind) == 0:
            continue
        cov_sub = cov[ind.reshape(-1, 1), ind.reshape(1, -1)]
        if torch.linalg.matrix_rank(cov_sub) == cov_sub.shape[0]:
            cov_vec = cov[ind, i].reshape(-1)
            bi = torch.linalg.solve(cov_sub, cov_vec)
            B[i, range(len(ind))] = bi
            ind_list[i, range(len(ind))] = ind
            F[i] = F[i] - torch.inner(cov_vec, bi)

    I_B = Sparse_B(torch.concatenate([torch.ones((n, 1)), -B], axis=1),
                   np.concatenate([np.arange(0, n).reshape(n, 1), ind_list], axis=1))

    return I_B, F #### Shall we return NNGP_cov instead?


def make_bf(coord: torch.Tensor,  #### could add a make_bf from cov (resolved)
            theta: tuple[float, float, float],
            neighbor_size: Optional[int] = 20,
            ) -> Tuple[Sparse_B, torch.Tensor]:
    """Obtain NNGP approximation with implicit covariance matrix

    Find the upper triangular matrix B and diagonal matrix F such that (I-B)'F^{-1}(I-B) appriximate the precision matrix
    (inverse of the covariance matrix). (see https://arxiv.org/abs/2102.13299 for more details.) Here only coordinates and
    spatial parameters are needed to represent the exponential covariance implicitly,
    thus being more memory-efficient than make_bf_from_cov.

    Parameters:
        coord:
            The nxd covariate array.
        neighbor_size:
            The number of nearest neighbors used used for NNGP approximation. Default being 20.
        theta:
            theta[0], theta[1], theta[2] represent sigma^2, phi, tau in the exponential covariance family.

    Returns:
        I_B:
            The sparse representation of I-B.
        F:
            The vector representation of the diagonal matrix.

    See Also:
        make_bf_from_cov : Obtain NNGP approximation of a covariance matrix \

        Datta, Abhirup. "Sparse nearest neighbor Cholesky matrices in spatial statistics."
        arXiv preprint arXiv:2102.13299 (2021).
    """
    n = coord.shape[0]
    rank = make_rank(coord, neighbor_size)
    B = torch.zeros((n, neighbor_size))
    ind_list = np.zeros((n, neighbor_size)).astype(int) - 1
    F = torch.zeros(n)
    for i in range(n):
        F[i] = make_cov_full(torch.tensor([0]), theta, nuggets = True)
        ind = rank[i, :][rank[i, :] <= i]
        if len(ind) == 0:
            continue
        cov_sub = make_cov_full(distance(coord[ind, :], coord[ind, :]), theta, nuggets = True)
        if torch.linalg.matrix_rank(cov_sub) == cov_sub.shape[0]:
            cov_vec = make_cov_full(distance(coord[ind, :], coord[i, :]), theta).reshape(-1)
            #### nuggets is not specified since its off-diagonal
            bi = torch.linalg.solve(cov_sub, cov_vec)
            B[i, range(len(ind))] = bi
            ind_list[i, range(len(ind))] = ind
            F[i] = F[i] - torch.inner(cov_vec, bi)

    I_B = Sparse_B(torch.concatenate([torch.ones((n, 1)), -B], axis=1),
                   np.concatenate([np.arange(0, n).reshape(n, 1), ind_list], axis = 1))

    return I_B, F

'''
def make_bf_np(coord: np.ndarray,  #### could add a make_bf from cov (resolved)
                neighbor_size: int,
                theta: tuple[float, float, float]
                ) -> Sparse_B:
    """The numpy version of make_bf

    Find the upper triangular matrix B and diagonal matrix F such that (I-B)'F^{-1}(I-B) appriximate the precision matrix
    (inverse of the covariance matrix). (see https://arxiv.org/abs/2102.13299 for more details.) Here only coordinates and
    spatial parameters are needed to represent the exponential covariance implicitly, 
    thus being more memory-efficient than make_bf_from_cov.

    Parameters
    ----------
    coord
        The nxd coordinate array.
    neighbor_size
        The number of nearest neighbors used used for NNGP approximation.
    theta
        theta[0], theta[1], theta[2] represent sigma^2, phi, tau in the exponential covariance family.

    Returns
    -------
    I_B: Sparse_B
        The sparse representation of I-B.
    F: torch.Tensor
        The vector representation of the diagonal matrix.

    See Also
    --------
    make_bf_from_cov : Obtain NNGP approximation of a covariance matrix
    make_bf : Obtain NNGP approximation with implicit covariance matrix
    """
    n = coord.shape[0]
    rank = make_rank(coord, neighbor_size)
    B = np.zeros((n, neighbor_size))
    ind_list = np.zeros((n, neighbor_size)).astype(int) - 1
    F = np.zeros(n)
    for i in range(n):
        F[i] = make_cov_full(theta, np.array([0]), nuggets = True)
        ind = rank[i, :][rank[i, :] <= i]
        if len(ind) == 0:
            continue
        cov_sub = make_cov_full(theta, distance_np(coord[ind, :], coord[ind, :]), nuggets = True)
        if np.linalg.matrix_rank(cov_sub) == cov_sub.shape[0]:
            cov_vec = make_cov_full(theta, distance_np(coord[ind, :], coord[i, :].reshape(1, -1))).reshape(-1) #### nuggets is not specified since its off-diagonal
            bi = np.linalg.solve(cov_sub, cov_vec)
            B[i, range(len(ind))] = bi
            ind_list[i, range(len(ind))] = ind
            F[i] = F[i] - np.inner(cov_vec, bi)

    I_B = Sparse_B(np.concatenate([np.ones((n, 1)), -B], axis=1),
                   np.concatenate([np.arange(0, n).reshape(n, 1), ind_list], axis = 1))

    return I_B, F
'''


def make_cov_full(dist: torch.Tensor | np.ndarray,
                  theta: tuple[float, float, float],
                  nuggets: Optional[bool] = False,
                  ) -> torch.Tensor | np.ndarray:
    """Compose covariance matrix from the distance matrix with dense representation.

    Compose a covariance matrix in the exponential covariance family (other options to be implemented) from the distance
    matrix. The returned object class depends on the input distance matrix.

    Parameters:
        dist:
            The nxn distance matrix
        theta:
            theta[0], theta[1], theta[2] represent sigma^2, phi, tau in the exponential covariance family.
        nuggets:
            Whether to include nuggets term in the covariance matrix (added to the diagonal).

    Returns:
        cov:
            A covariance matrix.
    """
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    if isinstance(dist, float) or isinstance(dist, int):
        dist = torch.Tensor(dist)
        n = 1
    else:
        n = dist.shape[-1]
    if isinstance(dist, torch.Tensor):
        cov = sigma_sq * torch.exp(-phi * dist)
    else:
        cov = sigma_sq * np.exp(-phi * dist)
    if nuggets:
        shape_temp = list(cov.shape)[:-2] + [1 ,1]
        if isinstance(dist, torch.Tensor):
            cov += tau_sq * torch.eye(n).repeat(*shape_temp).squeeze()
        else:
            cov += tau_sq * np.eye(n).squeeze() #### need improvement
    return cov

def make_cov(coord: torch.Tensor,
             theta: tuple[float, float, float],
             NNGP: Optional[bool] = True,
             neighbor_size: int = 20
             ) -> torch.Tensor:
    """Compose covariance matrix.

    Compose a covariance matrix in the exponential covariance family using the coordinates and spatial parameters.
    NNGP approximation is introduced for efficient representation. (see https://arxiv.org/abs/2102.13299 for more details.)

    Parameters:
        coord:
            The nxd covariate array.
        theta:
            theta[0], theta[1], theta[2] represent sigma^2, phi, tau in the exponential covariance family.
        NNGP:
            Whether use NNGP approximation (recommended and used by default).
        neighbor_size:
            Number of nearest neighbors used for NNGP approximation, default value is 20.

    Returns:
        cov:
            A covariance matrix as torch.Tensor (dense representation) or NNGP_cov (sparse representation).

    See Also:
        Datta, Abhirup. "Sparse nearest neighbor Cholesky matrices in spatial statistics."
        arXiv preprint arXiv:2102.13299 (2021).
    """
    if not NNGP:
        dist = distance(coord, coord)
        cov = make_cov_full(dist, theta, nuggets = True) #### could add a make_bf from cov (resolved)
        return cov
    else:
        I_B, F_diag = make_bf(coord, theta, neighbor_size) #### could merge into one step
        cov = NNGP_cov(I_B.B, F_diag, I_B.Ind_list)
        return cov

def Simulation(n: int, p:int,
               neighbor_size: int,
               fx: Callable,
               theta: tuple[float, float, float],
               coord: Optional[torch.tensor] = None,
               range: tuple[float, float] = [0,1],
               sparse: Optional[bool] = True
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | NNGP_cov, torch.Tensor]:
    """Simulate spatial data

    Simulate the spatial data based on the following model: Y(s) = f(X(s)) + w(s) + delta(s),
    where s are the spatial locations, X(s) is the spatial covariates, w(s) is the spatial effect (correlated noise),
    and delta(s) is the i.i.d random noise (nuggets).

    Parameters:
        n:
            Sample size.
        p:
            Dimension of covariates.
        fx:
            Function for the covariates' effect.
        theta:
            theta[0], theta[1], theta[2] represent sigma^2, phi, tau in the exponential covariance family,
            used for generating spatial random effect.
        coord:
            A nxd tensor as the locations where to simulate the data, if not specified, randomly sample from [range[0], range[1]]^2
            square.
        range:
            A tuple [a, b] as the range of spatial locations. The spatial coordinates are sampled uniformly from [a, b]^2.
        sparse:
            To be implemented. Mainly interact with the rmvn() function.

    Returns:
        X: torch.Tensor
            nxp array sampled uniformly from [0,1] as the covariates.
        Y: torch.Tensor
            Length n vector as the observations consists of fixed covariate effect, spatial random effect, and random noise.
        coord: torch.Tensor
            Simulated spatial locations.
        cov:
            Covariance matrix based on the simulated coordinates.
        corerr:
            Simulated spatial random effect.
    """
    if coord is None:
        coord = (range[1] - range[0]) * torch.rand(n, 2) + range[0]
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq

    cov = make_cov(coord, theta, neighbor_size)
    X = torch.rand(n, p)
    corerr = rmvn(torch.zeros(n), cov, sparse)
    Y = fx(X).reshape(-1) + corerr + torch.sqrt(tau_sq) * torch.randn(n)

    return X, Y, coord, cov, corerr

def spatial_order(X: torch.Tensor,
                  Y: torch.Tensor,
                  coord: torch.Tensor,
                  method = 'max-min',
                  numpropose = 2
                  ) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """Spatial ordering for data

    Order the data according to their spatial locations. A spatial ordering is necessary for NNGP to represent a valid
    spatial process. (Datta et.al 2016)
    Method 'coord-sum' stands for a simple spatial ordering by the summation of coordinates.
    Method 'max-min' stands for the max-min ordering based on Euclidean distance. "Basically, this ordering starts at a
    point in the center, then adds points one at a time that maximize the minimum distance from all previous points in
    the ordering." (Katzfuss & Guinness 2021)

    Parameters:
        X:
            nxp array of the covariates.
        Y:
            Length n vector as the response.
        coord:
            nxd array of the coordinates
        method:
            Method 'coord-sum' stands for a simple spatial ordering by the summation of coordinates.
            Method 'max-min' stands for the max-min ordering based on Euclidean distance. (Katzfuss & Guinness 2021)

    Returns:
        Ordered X, Ordered Y, Ordered coordinates, order

    See Also:
        Datta, Abhirup, et al. "Hierarchical nearest-neighbor Gaussian process models for large geostatistical datasets."
        Journal of the American Statistical Association 111.514 (2016): 800-812. \
        Katzfuss, Matthias & Guinness, Joseph. "A General Framework for Vecchia Approximations of Gaussian Processes."
        Statist. Sci. 36 (1) 124 - 141, February 2021. https://doi.org/10.1214/19-STS755
    """
    n = coord.shape[0]
    if n >= 10000 and method == 'max-min':
        warnings.warn("Too large for max-min ordering, switch to 'coord-sum' ordering!")
        method = 'coord-sum'

    d = coord.shape[1]
    if method == 'max-min':
        remaininginds = list(range(n))
        orderinds = torch.zeros(n, dtype=torch.int)
        distmp = distance(coord, coord.mean(dim = 0)).reshape(-1)
        ordermp = torch.argsort(distmp)
        orderinds[0] = ordermp[0]
        remaininginds.remove(orderinds[0])
        for j in range(1,n):
            randinds = random.sample(remaininginds, min(numpropose, len(remaininginds)))
            distarray = distance(coord[orderinds[0:j],:], coord[torch.tensor(randinds),:])
            bestind = torch.argmax(distarray.min(axis = 1).values)
            orderinds[j] = randinds[bestind]
            remaininginds.remove(orderinds[j])
    elif method == 'coord-sum':
        orderinds = torch.argsort(coord.sum(axis = 1))
    else:
        warnings.warn("Keep the order")
        orderinds = torch.tensor(range(n))

    return X[orderinds,:], Y[orderinds], coord[orderinds,:], orderinds


def make_graph(X: torch.Tensor,
               Y: torch.Tensor,
               coord: torch.Tensor,
               neighbor_size: Optional[int] = 20,
               Ind_list: Optional = None
               ) -> torch_geometric.data.Data:
    """Compose the data with graph information to work on.

    This function connects each node to its nearest neighbors and records the edge indexes in two forms for the downstream
    graph operations.

    Parameters:
        X:
            nxp array of the covariates.
        Y:
            Length n vector as the response.
        coord:
            nxd array of the coordinates
        neighbor_size:
            The number of nearest neighbors used for NNGP approximation. Default being 20.
        Ind_list:
            An optional index list. If provided, ommit the make_rank() step in the function.

    Returns:
        data: torch_geometric.data.Data \
            Data that can be processed by the torch_geometric framework.\
            data.x contains the covariates array,\
            data.y contains the response vector,\
            data.pos contains the spatial coordinates,\
            data.edge_index contains the indexes of form [i,j] where location j is in the nearest neighbor of location i.\
            data.edge_attr contains the concatenated numbering of the neighbors.\
            For each location, the numbering is of the form [0, 1, ... , k] where k is the number of the nearest neighbors.\
            This attribute is mainly used in messaging passing.

    See Also:
    make_rank : Compose the nearest neighbor index list based on the coordinates. \
    torch_geometric.data.Data : A data object describing a homogeneous graph.
    """
    n = X.shape[0]
    # Compute the edges of the graph
    edges = []
    neighbor_idc = []
    # Initialize the edges, the edges are predefined for the first m + 1 points
    # Find the m nearest neighbors for each remaining point
    if Ind_list is None:
        Ind_list = make_rank(coord, neighbor_size)
    for i in range(1, n):
        for j, idx in enumerate(Ind_list[i]):
            if idx < i:
                edges.append([idx, i])
                neighbor_idc.append(j)
            elif j >= i:
                break

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(neighbor_idc).reshape(-1, 1)  # denotes the index of the neighbor
    data = torch_geometric.data.Data(x=X, y=Y, pos=coord, edge_index=edge_index, edge_attr=edge_attr)
    assert data.validate(raise_on_error=True)
    return data

def split_data(X: torch.Tensor,
               Y: torch.Tensor,
               coord: torch.Tensor,
               neighbor_size: Optional[int] = 20,
               val_proportion: float = 0.2,
               test_proportion: float = 0.2
               ) -> tuple[torch_geometric.data.Data, torch_geometric.data.Data, torch_geometric.data.Data]:
    """Split the data into training, validation, and testing parts and add the graph information respectively.

    This function split the data with a user specified proportions and build the graph information. The output are the
    data sets that can be directly processed within the torch_geomtric framework.

    Parameters:
        X:
            nxp array of the covariates.
        Y:
            Length n vector as the response.
        coord:
            nxd array of the coordinates
        neighbor_size:
            The number of nearest neighbors used for NNGP approximation. Default being 20.
        val_proportion:
            The proportion of training set splitted as validation set.
        test_proportion:
            The proportion of whole data splitted as testing set.

    Returns:
        data_train, data_val, data_test

    See Also:
        make_graph : Compose the data with graph information to work on. \
        torch_geometric.data.Data : A data object describing a homogeneous graph.
    """
    X_train_val, X_test, Y_train_val, Y_test, coord_train_val, coord_test = train_test_split(
        X, Y, coord, test_size = test_proportion
    )

    X_train, X_val, Y_train, Y_val, coord_train, coord_val = train_test_split(
        X_train_val, Y_train_val, coord_train_val, test_size = val_proportion/(1 - test_proportion)
    )

    data_train = make_graph(X_train, Y_train, coord_train, neighbor_size)
    data_val = make_graph(X_val, Y_val, coord_val, neighbor_size)
    data_test = make_graph(X_test, Y_test, coord_test, neighbor_size)

    return data_train, data_val, data_test

def split_loader(data: torch_geometric.data.Data,
                 batch_size: Optional[int] = None
                 ) -> torch.DataLoaders:
    """Create mini-batches for GNN training

    This functions further split a data for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible. Note that only source nodes are splited into batches,
    each batch will contain edges originate from those source nodes.
    There might be interactions among the target nodes across batches.

    Parameters:
        data:
            Data with graph information (output of split_data() or make_graph()).
        batch_size:
            Size of mini-batches, default value being n/20.

    Returns:
        loader:
            A dataloader that can be enumerated for mini-batch training.

    See Also:
        make_graph : Compose the data with graph information to work on. \
        split_data : Split the data into training, validation, and testing parts and add the graph information respectively. \
        torch_geometric.loader : A data loader that performs neighbor sampling as introduced in the \

    Hamilton, Will, Zhitao Ying, and Jure Leskovec. "Inductive representation learning on large graphs."
    Advances in neural information processing systems 30 (2017).
    """
    if batch_size is None:
        batch_size  = int(data.x.shape[0]/20)
    loader = NeighborLoader(data,
                            input_nodes=torch.tensor(range(data.x.shape[0])),num_neighbors=[-1],
                            batch_size=batch_size, replace=False, shuffle=True)
    return loader

def edit_batch(batch): #### Change to a method
    edge_list = list()
    for i in range(batch.x.shape[0]):
        edge_list.append(batch.edge_attr[torch.where(batch.edge_index[1, :] == i)].squeeze())
    batch.edge_list = edge_list
    return batch

def theta_update(theta0: torch.Tensor,
                 w: torch.Tensor,
                 coord: torch.Tensor,
                 neighbor_size: Optional[int] = 20,
                 BRISC: Optional[bool] = True
                 ) -> np.array:
    """Update the spatial parameters using maximum likelihood.

    This function updates the spatial parameters by assuming the observations are from a Gaussian Process with exponential
    covariance function. Spatial coordinates and initial values of theta are input for building the covariance.
    By default, L-BFGS-B algorithm is used to optimize the likelihood.
    Since the likelihood is computed repeatedly, NNGP approximation is used for efficient computation of the log-likelihood,
    with a default neighbor size being 20.
    Note that we currently use the Cpp implementation of L-BFGS-B from the R-package BRISC (Saha & Datta 2018) as default,
    which is faster than the python version.
    In the future, we will introduce a python wrapper for the Cpp implementation to get free of R component.

    Parameters:
        theta0:
            Initial values of the spatial parameters.
            theta[0], theta[1], theta[2] represent sigma^2, phi, tau in the exponential covariance family.
        w:
            Length n observations of the spatial random effect without any fixed effect.
        coord:
            nx2 spatial coordinates of the observations.
        neighbor_size:
            The number of nearest neighbors used for NNGP approximation. Default being 20.
        BRISC:
            Whether to use the optimization from BRISC. Default being True. Setting as False will largely increase the running time.

    Returns:
        theta_updated:
            An updated tuple of the spatial paramaters.

    See Also:
        Datta, Abhirup, et al. "Hierarchical nearest-neighbor Gaussian process models for large geostatistical datasets."
        Journal of the American Statistical Association 111.514 (2016): 800-812.

        Zhu, Ciyou, et al. "Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization."
        ACM Transactions on mathematical software (TOMS) 23.4 (1997): 550-560.

        Saha, Arkajyoti, and Abhirup Datta. "BRISC: bootstrap for rapid inference on spatial covariances."
        Stat 7.1 (2018): e184.
    """
    warnings.filterwarnings("ignore")
    w = w.detach().numpy()
    coord = coord.detach().numpy()
    theta = theta0.detach().numpy()
    print('Theta updated from')
    print(theta)

    if not BRISC:
        n_train = w.shape[0]
        rank = make_rank(coord, neighbor_size)
        if n_train <= 2000:
            dist = distance_np(coord, coord)

            def likelihood(theta):
                sigma, phi, tau = theta
                cov = sigma * (np.exp(-phi * dist)) + tau * sigma * np.eye(n_train)  # need dist, n

                term1 = 0
                term2 = 0
                for i in range(n_train):
                    ind = rank[i, :][rank[i, :] <= i]
                    id = np.append(ind, i)

                    sub_cov = cov[ind, :][:, ind]
                    if np.linalg.det(sub_cov):
                        bi = np.linalg.solve(cov[ind, :][:, ind], cov[ind, i])
                    else:
                        bi = np.zeros(ind.shape)
                    I_B_i = np.append(-bi, 1)
                    F_i = cov[i, i] - np.inner(cov[ind, i], bi)
                    decor_res = np.sqrt(np.reciprocal(F_i)) * np.dot(I_B_i, w[id])
                    term1 += np.log(F_i)
                    term2 += decor_res ** 2
                return (term1 + term2)

        else:
            def likelihood(theta):
                sigma, phi, tau = theta

                term1 = 0
                term2 = 0
                for i in range(n_train):
                    ind = rank[i, :][rank[i, :] <= i]
                    if len(ind) == 0:
                        F_i = (1 + tau) * sigma
                        term1 += np.log(F_i)
                        term2 += w[i] ** 2 / F_i
                        continue

                    id = np.append(ind, i)

                    sub_dist = distance_np(coord[ind, :], coord[ind, :])
                    sub_dist_vec = distance_np(coord[ind, :], coord[i, :].reshape(1, -1)).reshape(-1)
                    sub_cov = sigma * (np.exp(-phi * sub_dist)) + tau * sigma * np.eye(len(ind))
                    sub_cov_vec = sigma * (np.exp(-phi * sub_dist_vec))
                    if np.linalg.det(sub_cov):
                        bi = np.linalg.solve(sub_cov, sub_cov_vec)
                    else:
                        bi = np.zeros(ind.shape)
                    I_B_i = np.append(-bi, 1)
                    F_i = (1 + tau) * sigma - np.inner(sub_cov_vec, bi)
                    decor_res = np.sqrt(np.reciprocal(F_i)) * np.dot(I_B_i, w[id])
                    term1 += np.log(F_i)
                    term2 += decor_res ** 2
                return (term1 + term2)

        res = scipy.optimize.minimize(likelihood, theta, method = 'L-BFGS-B',
                                      bounds = [(0, None), (0, None), (0, None)])
        return res.x

    else:
        _, theta = BRISC_estimation(w, None, coord)
        return theta

def krig_pred(w_train: torch.Tensor,
              coord_train: torch.Tensor,
              coord_test: torch.Tensor,
              theta: tuple[float, float, float],
              neighbor_size: Optional[int] = 20,
              q: Optional[float] = 0.95
              ) -> torch.Tensor:
    """Kriging prediction (Gaussian process regression) with confidence interval.

    Kriging prediction on testing locations based on the observations on the training locations. The kriging procedure
    assumes the observations are sampled from a Gaussian process, which is paramatrized here to have an exponential covariance
    structure using theta = [sigma^2, phi, tau]. NNGP appriximation is involved for efficient computation of matrix inverse.
    The conditional variance (kriging variance) is used to build the confidence interval using the quantiles (a/2, 1-a/2).
    (see https://arxiv.org/abs/2304.09157, section 4.3 for more details.)

    Parameters:
        w_train:
            Training observations of the spatial random effect without any fixed effect.
        coord_train:
            Spatial coordinates of the training observations.
        coord_test:
            Spatial coordinates of the locations for prediction
        theta:
            theta[0], theta[1], theta[2] represent sigma^2, phi, tau in the exponential covariance family.
        neighbor_size:
            The number of nearest neighbors used for NNGP approximation. Default being 20.
        q:
            Confidence coverage for the prediction interval. Default being 0.95.

    Returns:
        w_test: torch.Tensor
            The kriging prediction.
        pred_U: torch.Tensor
            Confidence upper bound.
        pred_L: torch.Tensor
            Confidence lower bound.

    See Also:
        Zhan, Wentao, and Abhirup Datta. 2024. “Neural Networks for Geospatial Data.”
        Journal of the American Statistical Association, June, 1–21. doi:10.1080/01621459.2024.2356293.
    """
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    n_test = coord_test.shape[0]

    rank = make_rank(coord_train, neighbor_size, coord_test)

    w_test = torch.zeros(n_test)
    sigma_test = (sigma_sq + tau_sq) * torch.ones(n_test)
    for i in range(n_test):
        ind = rank[i, :]
        cov_sub = make_cov_full(distance(coord_train[ind, :], coord_train[ind, :]), theta, nuggets=True)
        cov_vec = make_cov_full(distance(coord_train[ind, :], coord_test[i, :]), theta, nuggets=False).reshape(-1)
        bi = torch.linalg.solve(cov_sub, cov_vec)
        w_test[i] = torch.dot(bi.T, w_train[ind]).squeeze()
        sigma_test[i] = sigma_test[i] - torch.dot(bi.reshape(-1), cov_vec)
    p = scipy.stats.norm.ppf((1 + q) / 2, loc=0, scale=1)
    sigma_test = torch.sqrt(sigma_test)
    pred_U = w_test + p * sigma_test
    pred_L = w_test - p * sigma_test

    return w_test, pred_U, pred_L







        
