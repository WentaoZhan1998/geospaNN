import torch
import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors
from typing import Callable, Optional
from sklearn.model_selection import train_test_split
import torch_geometric
from torch_geometric.loader import NeighborLoader
import warnings
from torch_geometric.nn import MessagePassing

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
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

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class Sparse_B():
    def __init__(self, B, Ind_list):
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
                result[k] = torch.dot(self.B[i,range(len(ind))].reshape(-1),X[ind])
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
                x = torch.cat((x, y[i].unsqueeze(0)), dim = -1).float()
            else:
                x = torch.cat((x, (y[i] + torch.dot(x[ind[id]], B[i, :][id])).unsqueeze(0)), dim = -1).float()
        return x

    def Fmul(self, F):
        temp = Sparse_B(self.B.copy(), self.Ind_list.copy())
        for i in range(self.n):
            temp.B[i,:] = F[i]*self.B[i,:]
        return temp

    def to_dense(self):
        B = np.zeros((self.n, self.n))
        for i in range(self.n):
            ind = self.Ind_list[i, :][self.Ind_list[i, :] >= 0]
            if len(ind) == 0:
                continue
            B[i, ind] = self.B[i,range(len(ind))]
        return B

class NNGP_cov(Sparse_B):
    def __init__(self, B, F_diag, Ind_list):
        super().__init__(B = B, Ind_list = Ind_list)
        assert len(F_diag) == B.shape[0]
        self.F_diag = F_diag

    def correlate(self, x):
        assert x.shape[0] == self.n
        return self.invmul(torch.sqrt(self.F_diag) * x)

def rmvn(m: int,
         mu: torch.Tensor,
         cov: torch.Tensor | NNGP_cov,
         sparse: Optional[bool] = True):
    p = len(mu)
    if isinstance(cov, torch.Tensor):
        if p >= 2000: warnings.warn("Too large for cholesky decomposition, please try to use NNGP")
        D = torch.linalg.cholesky(cov)
        res = torch.matmul(torch.randn(m, p), D.t()) + mu
    elif isinstance(cov, NNGP_cov):
        if sparse:
            res = cov.correlate(torch.randn(m, p).reshape(-1))
        else:
            warnings.warn("To be implemented.")
    return  res.reshape(-1)

def make_rank(coord: torch.Tensor,
              neighbor_size: int,
              coord_test: Optional = None
              ) -> np.ndarray:
    if coord_test is None: neighbor_size += 1
    knn = NearestNeighbors(n_neighbors=neighbor_size)
    knn.fit(coord)
    if coord_test is None:
        coord_test = coord
        rank = knn.kneighbors(coord_test)[1]
        return rank[:, 1:]
    else:
        rank = knn.kneighbors(coord_test)[1]
        return rank[:, 0:]

def distance(coord1: torch.Tensor,
             coord2: torch.Tensor
             ) -> torch.Tensor:
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

def make_bf_from_cov(cov: torch.Tensor,
                     neighbor_size: int,
                     ) -> Sparse_B:
    n = cov.shape[0]
    B = torch.zeros((n, neighbor_size))
    ind_list = np.zeros((n, neighbor_size)).astype(int) - 1
    F = torch.zeros(n)
    rank = np.argsort(-cov, axis=-1)
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

    return I_B, F


def make_bf(coord: torch.Tensor,  #### could add a make_bf from cov (resolved)
            rank: np.ndarray,
            theta: tuple[float, float, float]
            ) -> Sparse_B:
    n = coord.shape[0]
    neighbor_size = rank.shape[1]
    B = torch.zeros((n, neighbor_size))
    ind_list = np.zeros((n, neighbor_size)).astype(int) - 1
    F = torch.zeros(n)
    for i in range(n):
        F[i] = make_cov_full(theta, torch.tensor([0]), nuggets = True)
        ind = rank[i, :][rank[i, :] <= i]
        if len(ind) == 0:
            continue
        cov_sub = make_cov_full(theta, distance(coord[ind, :], coord[ind, :]), nuggets = True)
        if torch.linalg.matrix_rank(cov_sub) == cov_sub.shape[0]:
            cov_vec = make_cov_full(theta, distance(coord[ind, :], coord[i, :])).reshape(-1) #### nuggets is not specified since its off-diagonal
            bi = torch.linalg.solve(cov_sub, cov_vec)
            B[i, range(len(ind))] = bi
            ind_list[i, range(len(ind))] = ind
            F[i] = F[i] - torch.inner(cov_vec, bi)

    I_B = Sparse_B(torch.concatenate([torch.ones((n, 1)), -B], axis=1),
                   np.concatenate([np.arange(0, n).reshape(n, 1), ind_list], axis = 1))

    return I_B, F

def make_cov_full(theta: tuple[float, float, float],
                  dist: torch.Tensor,
                  nuggets: Optional[bool] = False,
                  ) -> torch.Tensor:
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    if isinstance(dist, float) or isinstance(dist, int):
        dist = torch.Tensor(dist)
        n = 1
    else:
        n = dist.shape[-1]
    cov = sigma_sq * torch.exp(-phi * dist)
    if nuggets:
        shape_temp = list(cov.shape)[:-2] + [1 ,1]
        cov += tau_sq * torch.eye(n).repeat(*shape_temp).squeeze()
    return cov

def make_cov(theta: tuple[float, float, float],
             coord: torch.Tensor,
             NNGP: Optional[bool] = True,
             neighbor_size: int = 20
             ) -> torch.Tensor:
    dist = distance(coord, coord)

    if not NNGP:
        cov = make_cov_full(theta, dist, nuggets = True) #### could add a make_bf from cov (resolved)
        return cov
    else:
        rank = make_rank(coord, neighbor_size)
        I_B, F_diag = make_bf(coord, rank, theta) #### could merge into one step
        cov = NNGP_cov(I_B.B, F_diag, I_B.Ind_list)
        return cov

def Simulation(n: int, p:int,
               neighbor_size: int,
               fx: Callable,
               theta: tuple[float, float, float],
               range: tuple[float, float] = [0,1],
               sparse: Optional[bool] = True
               ):
    coord = (range[1] - range[0]) * torch.rand(n, 2) + range[0]
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq

    cov = make_cov(theta, coord, neighbor_size)
    X = torch.rand(n, p)
    corerr = rmvn(1, torch.zeros(n), cov, sparse)
    Y = fx(X).reshape(-1) + corerr + torch.sqrt(tau_sq) * torch.randn(n)

    return X, Y, coord, cov, corerr

def make_graph(X: torch.Tensor,
               Y: torch.Tensor,
               coord: torch.Tensor,
               neighbor_size: int,
               Ind_list: Optional = None
               ) -> torch_geometric.data.Data:
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
               batch_size: Optional[int] = None,
               neighbor_size: Optional[int] = 20,
               val_proportion: float = 0.2,
               test_proportion: float = 0.2
               ) -> tuple[torch_geometric.data.Data, torch_geometric.data.Data]:
    n = X.shape[0]
    if batch_size is None:
        batch_size  = int(n*(1 - test_proportion)/20)
    X_train_val, X_test, Y_train_val, Y_test, coord_train_val, coord_test = train_test_split(
        X, Y, coord, test_size = test_proportion
    )

    X_train, X_val, Y_train, Y_val, coord_train, coord_val = train_test_split(
        X_train_val, Y_train_val, coord_train_val, test_size = val_proportion/(1 - test_proportion)
    )

    data_train = make_graph(X_train, Y_train, coord_train, neighbor_size)
    data_val = make_graph(X_val, Y_val, coord_val, neighbor_size)
    data_test = make_graph(X_test, Y_test, coord_test, neighbor_size)

    train_loader = NeighborLoader(data_train, input_nodes=torch.tensor(range(data_train.x.shape[0])),
                                  num_neighbors=[-1], batch_size=batch_size, replace=False, shuffle=True)

    return train_loader, data_train, data_val, data_test

def edit_batch(batch): #### Change to a method
    edge_list = list()
    for i in range(batch.x.shape[0]):
        edge_list.append(batch.edge_attr[torch.where(batch.edge_index[1, :] == i)].squeeze())
    batch.edge_list = edge_list
    return batch

class GatherNeighborPositionsConv(MessagePassing):
    def __init__(self, neighbor_size, coord_dimensions):
        super().__init__(aggr="sum")
        self.neighbor_size = neighbor_size
        self.coord_dimensions = coord_dimensions

    def forward(self, pos, edge_index, edge_attr, batch_size):
        positions = self.propagate(edge_index, pos=pos, edge_attr=edge_attr)[range(batch_size), :]
        zero_index = torch.where(positions == 0)
        positions[zero_index] = torch.rand(zero_index[0].shape) * 10000 * (positions.max() - positions.min())
        return positions

    def message(self, pos_j, edge_attr):
        num_edges = edge_attr.shape[0]
        msg = torch.zeros(num_edges, self.neighbor_size * self.coord_dimensions)
        col_idc = edge_attr.flatten() * self.coord_dimensions
        row_idc = torch.tensor(range(num_edges))
        msg[
            row_idc.unsqueeze(1), col_idc.unsqueeze(1) + torch.tensor(range(self.coord_dimensions))
        ] = pos_j
        return msg

class GatherNeighborInfoConv1(MessagePassing):
    """
    The output of node i will be a tensor of shape (neighbor_size+1,) where the j-th row contains
    the output of the (j+1)-th neighbor of node i. The first row will contain the output of node i.
    Assumes that the outputs are already computed.
    """

    def __init__(self, neighbor_size):
        super().__init__(aggr="sum")
        self.neighbor_size = neighbor_size

    def forward(self, y, edge_index, edge_attr):
        out = self.propagate(edge_index, y = y.reshape(-1,1), edge_attr=edge_attr)
        out[:, 0] += y.squeeze()
        return out

    def message(self, y_j, edge_attr):
        num_edges = edge_attr.shape[0]
        msg = torch.zeros(num_edges, self.neighbor_size + 1)
        col_idc = edge_attr.flatten() + 1
        row_idc = torch.tensor(range(num_edges))
        msg[row_idc, col_idc] = y_j.squeeze()
        return msg

class GatherNeighborInfoConv(MessagePassing):
    """
    The output of node i will be a tensor of shape (neighbor_size+1,) where the j-th row contains
    the output of the (j+1)-th neighbor of node i. The first row will contain the output of node i.
    Assumes that the outputs are already computed.
    """

    def __init__(self, neighbor_size):
        super().__init__(aggr="sum")
        self.neighbor_size = neighbor_size

    def forward(self, y, edge_index, edge_attr, batch_size):
        out = self.propagate(edge_index, y = y.reshape(-1,1), edge_attr=edge_attr)[range(batch_size),:]
        return out

    def message(self, y_j, edge_attr):
        num_edges = edge_attr.shape[0]
        msg = torch.zeros(num_edges, self.neighbor_size).double()
        col_idc = edge_attr.flatten()
        row_idc = torch.tensor(range(num_edges))
        msg[row_idc, col_idc] = y_j.squeeze().double()
        return msg

class CovarianceVectorConv(MessagePassing):
    def __init__(self, neighbor_size, theta):
        super().__init__(aggr="sum")
        self.neighbor_size = neighbor_size
        self.theta = theta

    def forward(self, pos, edge_index, edge_attr, batch_size):
        return self.propagate(edge_index, pos = pos, edge_attr=edge_attr)[range(batch_size),:]

    def message(self, pos_i, pos_j, edge_attr):
        num_edges = edge_attr.shape[0]
        msg = torch.zeros(num_edges, self.neighbor_size)
        col_idc = edge_attr.flatten()
        row_idc = torch.tensor(range(num_edges))
        msg[row_idc, col_idc] = make_cov_full(self.theta, distance(pos_i - pos_j, torch.zeros(1,2))).squeeze()
        return msg

class InverseCovMatrixFromPositions(torch.nn.Module):
    def __init__(self, neighbor_size, coord_dimension, theta):
        super(InverseCovMatrixFromPositions, self).__init__()
        self.neighbor_size = neighbor_size
        self.coord_dimension = coord_dimension
        self.theta = theta

    def forward(self, neighbor_positions, edge_list):
        batch_size = neighbor_positions.shape[0]
        neighbor_positions = neighbor_positions.reshape(-1, self.neighbor_size, self.coord_dimension)
        neighbor_positions1 = neighbor_positions.unsqueeze(1)
        neighbor_positions2 = neighbor_positions.unsqueeze(2)
        dists = torch.sqrt(torch.sum((neighbor_positions1 - neighbor_positions2) ** 2, axis=-1))
        cov = make_cov_full(self.theta, dists, nuggets = True) #have to add nuggets (resolved)
        #cov_final = self.theta[0]*torch.eye(self.neighbor_size).repeat(batch_size, 1, 1)
        #for i in range(batch_size):
        #    cov_final[i, edge_list[i].reshape(1, -1, 1), edge_list[i].reshape(1, 1, -1)] = \
        #        cov[i, edge_list[i].reshape(1, -1, 1), edge_list[i].reshape(1, 1, -1)]
        #inv_cov_final = torch.linalg.inv(cov_final)
        inv_cov_final = torch.linalg.inv(cov)
        return inv_cov_final


class nngls(torch.nn.Module):
    def __init__(
        self,
        p: int,
        neighbor_size: int,
        coord_dimensions: int,
        mlp: torch.nn.Module,
        theta: tuple[float, float, float]
    ):
        super(nngls, self).__init__()
        self.p = p
        self.neighbor_size = neighbor_size
        self.coord_dimensions = coord_dimensions
        self.theta = torch.nn.Parameter(torch.Tensor(theta)) # split to accelerate?
        self.compute_covariance_vectors = CovarianceVectorConv(neighbor_size, self.theta)
        self.compute_inverse_cov_matrices = InverseCovMatrixFromPositions(
            neighbor_size, coord_dimensions, self.theta
        )
        self.gather_neighbor_positions = GatherNeighborPositionsConv(neighbor_size, coord_dimensions)
        self.gather_neighbor_outputs = GatherNeighborInfoConv(neighbor_size)
        self.gather_neighbor_targets = GatherNeighborInfoConv(neighbor_size)

        # Simple MLP to map features to scalars
        self.mlp = mlp

    def forward(self, batch):
        if 'batch_size' not in batch.keys:
            batch.batch_size = batch.x.shape[0] #### can improve
        batch = edit_batch(batch)
        Cov_i_Ni = self.compute_covariance_vectors(batch.pos, batch.edge_index, batch.edge_attr, batch.batch_size)
        coord_neighbor = self.gather_neighbor_positions(batch.pos, batch.edge_index, batch.edge_attr, batch.batch_size)
        Inv_Cov_Ni_Ni = self.compute_inverse_cov_matrices(coord_neighbor, batch.edge_list)

        B_i = torch.matmul(Inv_Cov_Ni_Ni, Cov_i_Ni.unsqueeze(2)).squeeze()
        F_i = self.theta[0] + self.theta[2] - torch.sum(B_i * Cov_i_Ni, dim = 1)

        y_neighbor = self.gather_neighbor_outputs(batch.y, batch.edge_index, batch.edge_attr, batch.batch_size)
        y_decor = (batch.y[range(batch.batch_size)] - torch.sum(y_neighbor * B_i, dim = 1))/torch.sqrt(F_i)
        batch.o = self.mlp(batch.x).squeeze()
        o_neighbor = self.gather_neighbor_outputs(batch.o, batch.edge_index, batch.edge_attr, batch.batch_size)
        o_decor = (batch.o[range(batch.batch_size)] - torch.sum(o_neighbor * B_i, dim=1)) / torch.sqrt(F_i)
        preds = batch.o

        return y_decor, o_decor, preds

    def estimate(self, X):
        with torch.no_grad():
            return self.mlp(X).squeeze()

    def predict(self, data_train, data_test, **kwargs):
        with torch.no_grad():
            w_train = data_train.y - self.estimate(data_train.x)
            w_test, _, _ = krig_pred(w_train, data_train.pos, data_test.pos, self.theta, **kwargs)
            estimation_test = self.estimate(data_test.x)
            return estimation_test + w_test


def krig_pred(w_train: torch.Tensor,
              coord_train: torch.Tensor,
              coord_test: torch.Tensor,
              theta: tuple[float, float, float],
              neighbor_size: Optional[int] = 20,
              q: Optional[float] = 0.95
              ) -> torch.Tensor:
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    n_test = coord_test.shape[0]

    rank = make_rank(coord_train, neighbor_size, coord_test=coord_test)

    w_test = torch.zeros(n_test)
    sigma_test = (sigma_sq + tau_sq) * torch.ones(n_test)
    for i in range(n_test):
        ind = rank[i, :]
        cov_sub = make_cov_full(theta, distance(coord_train[ind, :], coord_train[ind, :]), nuggets=True)
        cov_vec = make_cov_full(theta, distance(coord_train[ind, :], coord_train[i, :])).reshape(-1)
        bi = torch.linalg.solve(cov_sub, cov_vec)
        w_test[i] = torch.dot(bi.T, w_train[ind]).squeeze()
        sigma_test[i] = sigma_test[i] - torch.dot(bi.reshape(-1), cov_vec)
    p = scipy.stats.norm.ppf((1 + q) / 2, loc=0, scale=1)
    sigma_test = torch.sqrt(sigma_test)
    pred_U = w_test + p * sigma_test
    pred_L = w_test - p * sigma_test

    return w_test, pred_U, pred_L







        
