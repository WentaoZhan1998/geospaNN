#### 20230928
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import torch
import numpy as np
import scipy
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
import logging
import torch_sparse

# NNGP #################################################################################################################
def solve_I_B_sparseB(I_B, y):
    y = y.reshape(-1)
    n = y.shape[0]
    x = np.empty(0)
    Indlist = I_B.Ind_list[:,1:]
    B = -I_B.B[:,1:]
    for i in range(n):
        ind = Indlist[i,:]
        id = ind>=0
        if len(id) == 0:
            x = np.append(x,y[i])
            continue
        x = np.append(x, y[i] + np.dot(x[ind[id]], B[i,:][id]))
    return x

def rmvn(m, mu, cov, I_B, F_diag, sparse, chol = True):
    p = len(mu)
    if p <= 2000 and chol:
        D = np.linalg.cholesky(cov)
        res = np.matmul(np.random.randn(m, p), np.matrix.transpose(D)) + mu
    elif sparse:
        res = solve_I_B_sparseB(I_B, np.sqrt(F_diag) * np.random.randn(m, p).reshape(-1))
    else:
        res = scipy.linalg.solve_triangular(I_B, np.sqrt(F_diag) * np.random.randn(m, p).reshape(-1),lower=True)
    return  res.reshape(-1) # * np.ones((m, p))

def make_cov(theta, dist):
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    if isinstance(dist, float) or isinstance(dist, int):
        n = 1
    else:
        n = dist.shape[0]
    cov = sigma_sq * np.exp(-phi * dist) #+ tau_sq * np.eye(n)
    return (cov)

def make_rank(coord, nn, coord_test = None):
    knn = NearestNeighbors(n_neighbors=nn)
    knn.fit(coord)
    if coord_test is None:
        coord_test = coord
        rank = knn.kneighbors(coord_test)[1]
        return rank[:, 1:]
    else:
        rank = knn.kneighbors(coord_test)[1]
        return rank[:, 0:]

class Sparse_B():
    def __init__(self, B, Ind_list):
        self.B = B
        self.n = B.shape[0]
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
        return(result)

    def Fmul(self, F):
        temp = Sparse_B(self.B.copy(), self.Ind_list.copy())
        for i in range(self.n):
            temp.B[i,:] = F[i]*self.B[i,:]
        return(temp)

    def to_dense(self):
        B = np.zeros((self.n, self.n))
        for i in range(self.n):
            ind = self.Ind_list[i, :][self.Ind_list[i, :] >= 0]
            if len(ind) == 0:
                continue
            B[i, ind] = self.B[i,range(len(ind))]
        return(B)

def make_bf_dense(coord, rank, theta):
    n = coord.shape[0]
    k = rank.shape[1]
    B = np.zeros((n, n))
    F = np.zeros(n)
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    for i in range(n):
        F[i] = make_cov(theta, 0) + tau_sq
        ind = rank[i, :][rank[i, :] <= i]
        if len(ind) == 0:
            continue
        cov_sub = make_cov(theta, distance(coord[ind, :], coord[ind, :])) + tau_sq*np.eye(len(ind))
        if np.linalg.matrix_rank(cov_sub) == cov_sub.shape[0]:
            cov_vec = make_cov(theta, distance(coord[ind, :], coord[i, :])).reshape(-1)
            bi = np.linalg.solve(cov_sub, cov_vec)
            B[i, ind] = bi
            F[i] = F[i] - np.inner(cov_vec, bi)

    I_B = np.eye(n) - B
    return I_B, F

def make_bf_sparse(coord, rank, theta):
    n = coord.shape[0]
    k = rank.shape[1]
    B = np.zeros((n, k))
    ind_list = np.zeros((n, k)).astype(int) - 1
    F = np.zeros(n)
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    for i in range(n):
        F[i] = make_cov(theta, 0) + tau_sq
        ind = rank[i, :][rank[i, :] <= i]
        if len(ind) == 0:
            continue
        cov_sub = make_cov(theta, distance(coord[ind, :], coord[ind, :])) + tau_sq * np.eye(len(ind))
        if np.linalg.matrix_rank(cov_sub) == cov_sub.shape[0]:
            cov_vec = make_cov(theta, distance(coord[ind, :], coord[i, :])).reshape(-1)
            bi = np.linalg.solve(cov_sub, cov_vec)
            B[i, range(len(ind))] = bi
            ind_list[i, range(len(ind))] = ind
            F[i] = F[i] - np.inner(cov_vec, bi)

    I_B = Sparse_B(np.concatenate([np.ones((n, 1)), -B], axis=1),
                   np.concatenate([np.arange(0, n).reshape(n, 1), ind_list], axis=1))

    return I_B, F

def sparse_decor(coord, nn, theta):
    n = coord.shape[0]
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    rank = make_rank(coord, nn)
    F = np.zeros(n)
    row_indices = np.empty(0)
    col_indices = np.empty(0)
    values = np.empty(0)
    for i in range(n):
        F[i] = make_cov(theta, 0) + tau_sq
        ind = rank[i, :][rank[i, :] <= i]
        if len(ind) == 0:
            F[i] = np.sqrt(np.reciprocal(F[i]))
            continue
        cov_sub = make_cov(theta, distance(coord[ind, :], coord[ind, :])) + tau_sq * np.eye(len(ind))
        if np.linalg.matrix_rank(cov_sub) == cov_sub.shape[0]:
            cov_vec = make_cov(theta, distance(coord[ind, :], coord[i, :])).reshape(-1)
            bi = np.linalg.solve(cov_sub, cov_vec)
            F[i] = np.sqrt(np.reciprocal(F[i] - np.inner(cov_vec, bi)))
            row_indices = np.append(row_indices, np.repeat(i, len(ind)))
            col_indices = np.append(col_indices, ind)
            values = np.append(values, -bi * F[i])

    row_indices = np.append(row_indices, np.array(range(n)))
    col_indices = np.append(col_indices, np.array(range(n)))
    l = row_indices.shape[0]
    values = np.append(values, F)
    FI_B = [np.concatenate([row_indices.reshape(1, l), col_indices.reshape(1, l)], axis=0).astype(int), values]
    #FI_B = torch.sparse_csr_tensor(torch.from_numpy(row_indices).int(),
    #                            torch.from_numpy(col_indices).int(),
    #                            torch.from_numpy(values).float())

    return FI_B

def sparse_decor_sparseB(coord, nn, theta):
    n = coord.shape[0]
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    rank = make_rank(coord, nn)
    ind_list = np.zeros((n, nn)).astype(int) - 1
    B = np.zeros((n, nn))
    F = np.zeros(n)
    values = np.empty(0)
    for i in range(n):
        F[i] = make_cov(theta, 0) + tau_sq
        ind = rank[i, :][rank[i, :] <= i]
        if len(ind) == 0:
            F[i] = np.sqrt(np.reciprocal(F[i]))
            continue
        cov_sub = make_cov(theta, distance(coord[ind, :], coord[ind, :])) + tau_sq * np.eye(len(ind))
        if np.linalg.matrix_rank(cov_sub) == cov_sub.shape[0]:
            cov_vec = make_cov(theta, distance(coord[ind, :], coord[i, :])).reshape(-1)
            bi = np.linalg.solve(cov_sub, cov_vec)
            F[i] = np.sqrt(np.reciprocal(F[i] - np.inner(cov_vec, bi)))
            B[i, range(len(ind))] = bi * F[i]
            ind_list[i, range(len(ind))] = ind

    FI_B = Sparse_B(np.concatenate([(np.ones(n)*F).reshape((n,1)), -B], axis=1),
                   np.concatenate([np.arange(0, n).reshape(n, 1), ind_list], axis=1))

    return FI_B

def distance(coord1, coord2):
    if coord1.ndim == 1:
        m = 1
        p = coord1.shape[0]
        coord1 = coord1.reshape((1, p))
    else:
        m = coord1.shape[0]
    if coord2.ndim == 1:
        n = 1
        p = coord2.shape[0]
        coord2 = coord2.reshape((1, p))
    else:
        n = coord2.shape[0]

    dists = np.zeros((m, n))
    for i in range(m):
        dists[i, :] = np.sqrt(np.sum((coord1[i] - coord2) ** 2, axis=1))
    return(dists)

def bf_from_theta(theta, coord, nn, method = '0', nu = 1.5, sparse = True, version = 'sparseB'):
    n = coord.shape[0]
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    cov = 0
    if n<= 2000:
        dist = distance(coord, coord)
        if method == '0':
            cov = sigma_sq * np.exp(-phi * dist) + tau_sq * np.eye(n)
        elif method == '1':
            cov = sigma_sq * pow((dist * phi), nu) / (pow(2, (nu - 1)) * scipy.special.gamma(nu)) * \
                  scipy.special.kv(nu,dist * phi)
            cov[range(n), range(n)] = sigma_sq + tau_sq
    rank = make_rank(coord, nn)
    if sparse and version == 'sparseB':
        I_B, F_diag = make_bf_sparse(coord, rank, theta)
    else:
        I_B, F_diag = make_bf_dense(coord, rank, theta)
        I_B = torch.from_numpy(I_B)
    F_diag = torch.from_numpy(F_diag)

    return I_B, F_diag, rank, cov

# Data #################################################################################################################
def Simulate(n, p, fx, nn, theta, method = '0', nu = 1.5, a = 0, b = 1, sparse = True, meanshift = False):
    #n = 1000
    coord = np.random.uniform(low = a, high = b, size=(n, 2))
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq

    I_B, F_diag, rank, cov = bf_from_theta(theta, coord, nn, method = method, nu = nu, sparse = sparse)

    X = np.random.uniform(size=(n, p))
    corerr = rmvn(1, np.zeros(n), cov, I_B, F_diag, sparse)
    if meanshift:
        corerr = corerr - np.mean(corerr)

    Y = fx(X).reshape(-1) + corerr + np.sqrt(tau_sq) * np.random.randn(n)

    return X, Y, I_B, F_diag, rank, coord, cov, corerr

def Simulate_mis(n, p, fx, nn, corerr_gen, a=0, b=1):
    coord = np.random.uniform(low=a, high=b, size=(n, 2))
    corerr = corerr_gen(coord)
    rank = make_rank(coord, nn)
    X = np.random.uniform(size=(n, p))
    Y = fx(X).reshape(-1) + corerr

    return X, Y, rank, coord, corerr

def partition (list_in, n):
    idx = torch.randperm(list_in.shape[0])
    list_in = list_in[idx]
    return [torch.sort(list_in[i::n])[0] for i in range(n)]

def batch_gen (data, k):
    for mask in ['train_mask', 'val_mask', 'test_mask']:
        data[mask + '_batch'] = partition(torch.tensor(range(data.n))[data[mask]],
                                          int(torch.sum(data[mask])/k))
    return(data)
# Models ###############################################################################################################
class Netp_sig(torch.nn.Module):
    def __init__(self, p, k = 50, q = 1):
        super(Netp_sig, self).__init__()
        self.l1 = torch.nn.Linear(p, k)
        self.l2 = torch.nn.Linear(k, q)

    def forward(self, x, edge_index = 0):
        x = torch.sigmoid(self.l1(x))
        return self.l2(x)

class Netp_tanh(torch.nn.Module):
    def __init__(self, p, k = 50, q = 1):
        super(Netp_tanh, self).__init__()
        self.l1 = torch.nn.Linear(p, k)
        self.l2 = torch.nn.Linear(k, q)

    def forward(self, x, edge_index = 0):
        x = torch.tanh(self.l1(x))
        return self.l2(x)

# Stopping #############################################################################################################

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

# Functions ############################################################################################################

def f1(X): return 10 * np.sin(np.pi * X)

def fx_l(x): return 5*x + 2

def f5(X): return (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] +5*X[:,4])/6

def f15(X): return (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] + 5*X[:,4] +
                    3/(X[:,5]+1)/(X[:,6]+1) + 4*np.exp(np.square(X[:,7])) + 30*np.square(X[:,8])*X[:,9] +
                    5*(np.exp(np.square(X[:,10]))*np.sin(np.pi*X[:,11]) + np.exp(np.square(X[:,11]))*np.sin(np.pi*X[:,10])) +
                    10*np.square(X[:,12])*np.cos(np.pi*X[:,13]) + 20*np.square(X[:,14]))/6

# R ####################################################################################################################
def import_BRISC():
    BRISC = importr('BRISC')
    return BRISC

BRISC = import_BRISC()

def BRISC_estimation(residual, X, coord):
    residual_r = robjects.FloatVector(residual)
    coord_r = robjects.FloatVector(coord.transpose().reshape(-1))
    coord_r = robjects.r['matrix'](coord_r, ncol=2)

    if X is None:
        res = BRISC.BRISC_estimation(coord_r, residual_r)
    else:
        Xr = robjects.FloatVector(X.transpose().reshape(-1))
        Xr = robjects.r['matrix'](Xr, ncol=X.shape[1])
        res = BRISC.BRISC_estimation(coord_r, residual_r, Xr)

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

# Decorrelation ########################################################################################################
def decor_dense(y, FI_B_local, idx = None):
    if idx is None: idx = range(y.shape[0])
    y = y.reshape(-1)
    y_decor = torch.matmul(FI_B_local[idx,:].double(), y.double())
    return(y_decor.float())

def decor_sparse(y, FI_B_local, idx = None):
    n = y.shape[0]
    y = y.reshape((n, 1))
    y_decor = torch_sparse.spmm(torch.from_numpy(FI_B_local[0]), torch.from_numpy(FI_B_local[1]), n, n, y)
    #y_decor = torch.sparse.mm(FI_B_local, y)
    return(y_decor.reshape(-1))

def decor_dense_np(y, FI_B_local, idx=None):
    if idx is None: idx = range(y.shape[0])
    y_decor = np.matmul(FI_B_local[idx, :], y)
    return (y_decor)

def decor_sparse_np(y, FI_B_local, idx=None):
    if idx is None: idx = range(y.shape[0])
    n = len(idx)
    if np.ndim(y) == 2:
        p = y.shape[1]
        y_decor = np.zeros((n, p))
        for i in range(n):
            y_decor[i, :] = np.dot(FI_B_local.B[idx[i], :], y[FI_B_local.Ind_list[idx[i], :], :])
    elif np.ndim(y) == 1:
        y_decor = np.zeros(n)
        for i in range(n):
            y_decor[i] = np.dot(FI_B_local.B[idx[i], :], y[FI_B_local.Ind_list[idx[i], :]])
    return (y_decor)

def decor_sparse_SparseB(y, FI_B_local, idx = None):
    if idx is None: idx = range(y.shape[0])
    y = y.reshape(-1)
    n = len(idx)
    y_decor = torch.zeros(n)
    for i in range(n):
        y_decor[i] = torch.dot(FI_B_local.B[idx[i],:], y[FI_B_local.Ind_list[idx[i],:]])
    return(y_decor.float())

def undecor(y_decor, I_B_inv_local, F_diag_local):
    y = torch.matmul(I_B_inv_local,
                     torch.sqrt(F_diag_local.double()) * y_decor.double())
    return(y)

# Resample #############################################################################################################
def resample_fun_sparseB(residual, coord, nn, theta, regenerate=False):
    FI_B = sparse_decor_sparseB(coord, nn, theta)
    FI_B = FI_B.to_tensor()
    residual_decor = decor_sparse_SparseB(residual, FI_B).detach().numpy()
    if regenerate:
        residual_decor = np.std(residual_decor) * np.random.randn(1, residual_decor.shape[0])
    rank = make_rank(coord, nn)
    I_B, F_diag = make_bf_sparse(coord, rank, theta)
    idx = torch.randperm(residual_decor.shape[0])
    residual_decor = residual_decor[idx]  # *np.random.choice([-1,1], residual_decor.shape[0])
    res = solve_I_B_sparseB(I_B, np.sqrt(F_diag) * residual_decor.reshape(-1))
    return (res)


def resample_fun(residual, I_B, I_B_inv, F_diag, resample='shuffle', regenerate=False):
    FI_B = (I_B.T * torch.sqrt(torch.reciprocal(F_diag))).T
    residual_decor = decor_dense(residual, FI_B).detach().numpy()
    if regenerate:
        residual_decor = np.std(residual_decor) * np.random.randn(1, residual_decor.shape[0])
    if resample == 'shuffle':
        idx = torch.randperm(residual_decor.shape[0])
    elif resample == 'choice':
        idx = np.random.choice(residual_decor.shape[0], residual_decor.shape[0])
    residual_decor = torch.from_numpy(residual_decor[idx])  # *torch.from_numpy(np.random.choice([-1,1], residual_decor.shape[0]))
    return (undecor(residual_decor, I_B_inv, F_diag))

# Training #############################################################################################################
MSE = torch.nn.MSELoss(reduction='mean')
def train_gen_new(model, optimizer, data, epoch_num, loss_fn = MSE,
                  patience = 20, patience_half = 10):
    def train(model, data, idx):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        out = torch.reshape(out, (-1,))
        loss = loss_fn(out[idx], data.y[idx])
        loss.backward()
        optimizer.step()
        return loss
    @torch.no_grad()
    def test(model, data):
        model.eval()
        pred = model(data.x, data.edge_index)
        pred = torch.reshape(pred, (-1,))
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(np.square(pred[mask] - data.y[mask]).mean())
        return accs

    lr_scheduler = LRScheduler(optimizer, patience=patience_half, factor=0.5)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.00001)
    losses = []
    val_losses = []
    best_val_loss = final_test_loss = 100
    for epoch in range(1, epoch_num):
        for idx in data.train_mask_batch:
            loss = train(model, data, idx)
        train_loss, val_loss, tmp_test_loss = test(model, data)
        losses.append(train_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_loss = tmp_test_loss
        lr_scheduler(val_loss)
        early_stopping(val_loss)
        val_losses.append(val_loss)
        if early_stopping.early_stop:
            print('End at epoch' + str(epoch))
            break
    return epoch, val_losses, model

def train_decor_new(model, optimizer, data, epoch_num, theta_hat0, BF = None, sparse = None, sparseB = True,
                    loss_fn = MSE, nn = 20,
                    patience = 20, patience_half = 10,
                    Update=True, Update_method='optimization',
                    Update_init=50, Update_step=50, Update_bound=0.1,
                    Update_lr_ctrl=False
                    ):
    def train_decor(model, data, FI_B, idx, decor):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        out = torch.reshape(out, (-1,))
        loss = loss_fn(decor(out,  FI_B, idx), decor(data.y,  FI_B, idx))
        loss.backward()
        optimizer.step()
        return loss
    @torch.no_grad()
    def test(model, data):
        model.eval()
        pred = model(data.x, data.edge_index)
        pred = torch.reshape(pred, (-1,))
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(np.square(pred[mask] - data.y[mask]).mean())
        return accs
    lr_scheduler = LRScheduler(optimizer, patience=patience_half, factor=0.5)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.00001)
    losses = []
    val_losses = []
    best_val_loss = final_test_loss = 100

    n = data.n
    coord = data.coord
    Y = data.y
    X = data.x
    theta_hat = np.array(theta_hat0.copy())

    if sparse == None and n >= 10000: sparse = True
    if sparse and sparseB:
        print('Using sparseB family!')

    if sparse:
        decor = decor_sparse_SparseB if sparseB else decor_sparse
        sparse_decor_fun = sparse_decor_sparseB if sparseB else sparse_decor
        FI_B = sparse_decor_fun(coord, nn, theta_hat)
        if sparseB: FI_B = FI_B.to_tensor()
    else:
        decor = decor_dense
        if BF != None:
            I_B, F_diag = BF
            if Update:
                knn = NearestNeighbors(n_neighbors=nn)
                knn.fit(coord)
                rank = knn.kneighbors(coord)[1][:, 1:]
                logging.warning('Theta overwritten by the BF matrix as initial value, update parameters anyway!')
        else:
            I_B, F_diag, rank, _ = bf_from_theta(theta_hat, coord, nn, sparse=sparse)
        FI_B = (I_B.T*torch.sqrt(torch.reciprocal(F_diag))).T

    for epoch in range(1, epoch_num):
        for idx in data.train_mask_batch:
            loss_local = train_decor(model, data, FI_B, idx, decor)
        train_loss, val_loss, tmp_test_loss = test(model, data)
        losses.append(train_loss)
        Y_hat = model(X.float()).reshape(-1).double()

        # Parameter updating
        if (epoch >= Update_init) & (epoch % Update_step == 0) & Update:
            if Update_method == 'optimization':
                def test2(theta_hat_test):
                    sigma, phi, tau = theta_hat_test
                    tau_sq = sigma*tau

                    Y_hat_local = Y_hat
                    err = (Y_hat_local - Y).detach().numpy()
                    term1 = 0
                    term2 = 0
                    for i in range(n):
                        ind = rank[i, :][rank[i, :] <= i]
                        id = np.append(ind, i)

                        sub_cov = make_cov(theta_hat_test, distance(coord[ind, :], coord[ind, :])) + tau_sq*np.eye(len(ind))
                        sub_vec = make_cov(theta_hat_test, distance(coord[i, :], coord[ind, :])).reshape(-1)
                        if np.linalg.det(sub_cov):
                            bi = np.linalg.solve(sub_cov, sub_vec)
                        else:
                            bi = np.zeros(ind.shape)
                        I_B_i = np.append(-bi, 1)
                        F_i = sigma + tau_sq - np.inner(sub_vec, bi)
                        err_decor = np.sqrt(np.reciprocal(F_i)) * np.dot(I_B_i, err[id])
                        term1 += np.log(F_i)
                        term2 += err_decor ** 2
                    return (term1 + term2)

                def constraint1(x):
                    return x[2]

                def constraint2(x):
                    return x[0]

                cons = [{'type': 'ineq', 'fun': constraint1},
                        {'type': 'ineq', 'fun': constraint2}]

                res = minimize(test2, theta_hat, constraints=cons)
                theta_hat_new = res.x
            elif Update_method == 'BRISC':
                residual_temp = model(X).reshape(-1) - Y
                residual_temp = residual_temp.detach().numpy()
                _, theta_hat_new = BRISC_estimation(residual_temp, X.detach().numpy(), coord)

            print(theta_hat_new)
            if np.sum((theta_hat_new - theta_hat) ** 2) / np.sum((theta_hat) ** 2) < Update_bound:
                theta_hat = theta_hat_new
                if sparse: FI_B = sparse_decor_fun(coord, nn, theta_hat)
                else:
                    I_B, F_diag, rank, _ = bf_from_theta(theta_hat, coord, nn, sparse=sparse)
                    FI_B = (I_B.T*torch.sqrt(torch.reciprocal(F_diag))).T
                print('theta updated')
                if Update_lr_ctrl == True:
                    for g in optimizer.param_groups:
                        learning_rate = g['lr']
                    for g in optimizer.param_groups:
                        g['lr'] = 4 * learning_rate
                    early_stopping.counter = -patience_half * 2 - int(patience / 2)
            print(theta_hat)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        lr_scheduler(val_loss)
        early_stopping(val_loss)
        val_losses.append(val_loss)
        if early_stopping.early_stop:
            print('End at epoch' + str(epoch))
            break
    return epoch, val_losses, model
    #return theta_hat, val_losses, model

#### Evaluation #######################################################################################################
def RMSE(x,y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    n = x.shape[0]
    return(np.sqrt(np.sum(np.square(x-y))/n))

def krig_pred(model, X_train, X_test, Y_train, coord_train, coord_test, theta_hat0, q = 0.95):
    theta_hat = theta_hat0.copy()
    residual_train = Y_train - model(torch.from_numpy(X_train).float()).detach().numpy().reshape(-1)
    sigma_sq, phi, tau = theta_hat
    tau_sq = tau * sigma_sq
    n_test = coord_test.shape[0]

    rank = make_rank(coord_train, nn = 20, coord_test = coord_test)

    residual_test = np.zeros(n_test)
    sigma_test = (make_cov(theta_hat, 0) + tau_sq) * np.ones(n_test)
    for i in range(n_test):
        ind = rank[i,:]
        C_N = make_cov(theta_hat, distance(coord_train[ind, :], coord_train[ind, :]))
        C_N = C_N + tau_sq * np.eye(C_N.shape[0])
        C_Ni = make_cov(theta_hat, distance(coord_train[ind, :], coord_test[i, :]))
        bi = np.linalg.solve(C_N, C_Ni)
        residual_test[i] = np.dot(bi.T, residual_train[ind])
        sigma_test[i] =  sigma_test[i] - np.dot(bi.reshape(-1), C_Ni)
    p = scipy.stats.norm.ppf((1+q)/2, loc=0, scale=1)
    sigma_test = np.sqrt(sigma_test)
    del C_N
    del C_Ni
    pred = model(torch.from_numpy(X_test).float()).detach().numpy().reshape(-1) + residual_test
    pred_U = pred + p*sigma_test
    pred_L = pred - p*sigma_test

    return([pred, pred_U, pred_L])


def krig_pred_fullGP(model, X_train, X_test, Y_train, coord_train, coord_test, theta_hat0, q=0.95):
    theta_hat = theta_hat0.copy()
    residual_train = Y_train - model(torch.from_numpy(X_train).float()).detach().numpy().reshape(-1)
    sigma_sq, phi, tau = theta_hat
    tau_sq = tau * sigma_sq
    n_test = coord_test.shape[0]

    sigma_test = (make_cov(theta_hat, 0) + tau_sq) * np.ones(n_test)
    C_N = make_cov(theta_hat, distance(coord_train, coord_train))
    C_N = C_N + tau_sq * np.eye(C_N.shape[0])
    C_Ni = make_cov(theta_hat, distance(coord_train, coord_test))
    bi = np.linalg.solve(C_N, C_Ni)
    residual_test = np.dot(bi.T, residual_train).reshape(-1)
    p = scipy.stats.norm.ppf((1 + q) / 2, loc=0, scale=1)
    sigma_test = np.sqrt(sigma_test)
    pred = model(torch.from_numpy(X_test).float()).detach().numpy().reshape(-1) + residual_test
    pred_U = pred + p * sigma_test
    pred_L = pred - p * sigma_test

    return ([pred, pred_U, pred_L])

def RMSE_model(model, X, Y, mask, coord, theta_hat0):
    X_train = X[mask, :]
    Y_train = Y[mask]
    X_test = X[~(mask), :]
    Y_test = Y[~(mask)]
    coord_train = coord[mask, :]
    coord_test = coord[~(mask), :]
    pred = krig_pred(model, X_train, X_test, Y_train, coord_train, coord_test, theta_hat0)[0]
    return(RMSE(pred, Y_test)/RMSE(Y_test, np.mean(Y_test)))

