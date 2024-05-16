import torch
import geospaNN
import numpy as np
import time
import pandas as pd

def f5(X): return (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] +5*X[:,4])/6

sigma = 1
phi = 3
tau = 0.01
theta = torch.tensor([sigma, phi / np.sqrt(2), tau])

p = 5; funXY = f5

n = 1000
rand = 0
nn = 20
batch_size = 50

torch.manual_seed(2023 + rand)
X, Y, coord, cov, corerr = geospaNN.Simulation(n, p, nn, funXY, theta, range=[0, 10])
if False:
    np.random.seed(2023+rand)
    X, Y, I_B, F_diag, rank, coord, cov, corerr = Simulate(n, p, funXY, nn, theta.detach().numpy(), method=method, a=0,
                                                                 b=10, sparse = Sparse)
    corerr = torch.from_numpy(corerr)
    coord = torch.from_numpy(coord)
    id = range(int(n))
    print(BRISC_estimation(corerr[id].detach().numpy(), None, coord[id,:].detach().numpy())[1])
    theta_update(torch.tensor([1, 1.5, 0.01]), corerr[id], coord[id,:], neighbor_size = 20)
data = geospaNN.make_graph(X, Y, coord, nn)
data_train, data_val, data_test = geospaNN.split_data(X, Y, coord, neighbor_size=20,
                                             test_proportion=0.2)

mlp = torch.nn.Sequential(
    torch.nn.Linear(p, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
)
nn_model = geospaNN.nn_train(mlp, lr =  0.01, min_delta = 0.001)
training_log = nn_model.train(data_train, data_val, data_test)
theta0 = geospaNN.theta_update(torch.tensor([1, 1.5, 0.01]), mlp(data_train.x).squeeze() - data_train.y, data_train.pos, neighbor_size = 20)
mlp = torch.nn.Sequential(
    torch.nn.Linear(p, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
)
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp, theta=torch.tensor(theta0))
nngls_model = nngls_train(model, lr =  0.01, min_delta = 0.001)
training_log = nngls_model.train(data_train, data_val, data_test,
                                 Update_init = 100, Update_step = 10)
theta_hat = theta_update(torch.tensor(theta0),
                         model.estimate(data_train.x).squeeze() - data_train.y,
                         data_train.pos, neighbor_size = nn)

training_log = {'val_loss': [], 'pred_loss': [], 'sigma': [], 'phi': [], 'tau': []}

if False:
    theta0 = torch.tensor([sigma, phi / np.sqrt(2), tau])
    neighbor_size = nn
    theta = theta0.detach().numpy()
    residual = data_train.y - mlp(data_train.x).squeeze()
    coord = data_train.pos
    residual = residual.detach().numpy() + 0.1
    coord = coord.detach().numpy()
    n_train = residual.shape[0]
    dist = distance_np(coord, coord)
    rank = make_rank(coord, neighbor_size)
    print('Theta updated from')
    print(theta)
    def likelihood(theta):
        sigma, phi, tau = theta
        cov = sigma * (np.exp(-phi * dist)) + tau * np.eye(n_train)  # need dist, n

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
            I_Bi = np.append(-bi, 1)
            Fi = cov[i, i] - np.inner(cov[ind, i], bi)
            decor_res = np.sqrt(np.reciprocal(Fi)) * np.dot(I_Bi, residual[id])
            term1 += np.log(Fi)
            term2 += decor_res ** 2
        return (term1 + term2)

    i = 0
    def likelihood_k(theta):
        sigma, phi, tau = theta
        cov = sigma * (np.exp(-phi * dist)) + tau * np.eye(n_train)  # need dist, n

        term1 = 0
        term2 = 0

        ind = rank[i, :][rank[i, :] <= i]
        id = np.append(ind, i)

        sub_cov = cov[ind, :][:, ind]
        if np.linalg.det(sub_cov):
            bi = np.linalg.solve(cov[ind, :][:, ind], cov[ind, i])
        else:
            bi = np.zeros(ind.shape)
        I_Bi = np.append(-bi, 1)
        Fi = cov[i, i] - np.inner(cov[ind, i], bi)
        decor_res = np.sqrt(np.reciprocal(Fi)) * np.dot(I_Bi, residual[id])
        term1 += np.log(Fi)
        term2 += decor_res ** 2
        return (term1 + term2)

    h = 0.000000001
    (likelihood_k([theta[0] + h, theta[1], theta[2]]) - likelihood_k([theta[0], theta[1], theta[2]]))/h
    (likelihood_k([theta[0], theta[1], theta[2] + h]) - likelihood_k([theta[0], theta[1], theta[2]])) / h
    def grad(theta):
        sigma, phi, tau = theta
        n = dist.shape[0]
        cov = sigma * (np.exp(-phi * dist) + tau * np.eye(n_train))
        cov_deriv = np.stack([np.exp(-phi * dist),
                                    -dist*sigma*np.exp(-phi * dist),
                                    np.eye(n)], axis = 0)
        deriv = 0
        for i in range(n_train):
            ind = rank[i, :][rank[i, :] <= i]
            id = np.append(ind, i)

            if len(ind) > 0:
                bi = np.linalg.solve(cov[ind, :][:, ind], cov[ind, i])
                ci_deriv = cov_deriv[:, ind, i].reshape((-1, len(ind)))
                Ci_deriv = cov_deriv[:, :, ind][:, ind, :]
                F_deriv = cov_deriv[:, 0, 0] - 2 * np.dot(ci_deriv, bi) - np.dot(np.dot(Ci_deriv, bi), bi)
                res_temp = residual[ind]
            else:
                bi = np.zeros(ind.shape)
                ci_deriv = np.zeros(len(theta))
                F_deriv = cov_deriv[:, 0, 0]
                res_temp = 0
            I_Bi = np.append(-bi, 1)
            Fi = cov[i, i] - np.inner(cov[ind, i], bi)
            residual_decor = np.dot(I_Bi, residual[id])
            decor_res = np.sqrt(np.reciprocal(Fi)) * residual_decor
            deriv_temp = F_deriv/Fi - decor_res*np.sqrt(np.reciprocal(Fi))*(
                    F_deriv * residual_decor / Fi +
                    np.dot(ci_deriv,  res_temp)
            )
            deriv += deriv_temp
        return deriv

    def constraint1(x):
        return x[0]
    def constraint2(x):
        return x[1]
    def constraint3(x):
        return x[2]

    cons = [{'type': 'ineq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint3}]

'''
jac{callable, ‘2-point’, ‘3-point’, ‘cs’, bool}, optional
Method for computing the gradient vector. 
Only for CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr. 

boundssequence or Bounds, optional
Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, trust-constr, and COBYLA methods. 
'''

    start_time = time.time()
    res = scipy.optimize.minimize(likelihood, theta, method = 'BFGS',
                                  constraints=cons)
    print(res.x)
    print(time.time() - start_time)
    start_time = time.time()
    res = scipy.optimize.minimize(likelihood, theta, method = 'L-BFGS-B',
                                  bounds = [(0, None), (0, None), (0, None)])
    print(res.x)
    print(time.time() - start_time)

BRISC_estimation(residual, None, coord)

### Running time
if False:
    t = np.empty(0)
    size_vec = np.empty(0)
    epoch_vec = np.empty(0)

    for n in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
        for rand in range(1, 5 + 1):
            nn = 20
            batch_size = 50

            torch.manual_seed(2023 + rand)
            X, Y, coord, cov, corerr = Simulation(n, p, nn, funXY, theta, range = [0,1])
            data = make_graph(X, Y, coord, nn)
            data_train, data_val, data_test = split_data(X, Y, coord, neighbor_size = 20,
                                                             test_proportion = 0.2)
            train_loader = split_loader(data_train, batch_size)

            start_time = time.time()
            mlp = torch.nn.Sequential(
                        torch.nn.Linear(p, 5),
                        torch.nn.ReLU(),
                        torch.nn.Linear(5, 1)
                    )
            model = nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp = mlp, theta = torch.tensor([1.5, 5, 0.1]))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            Update_init = 0; Update_step = 1; Update_bound = 0.1; patience_half = 5; patience = 10;
            lr_scheduler = LRScheduler(optimizer, patience = patience_half, factor=0.5)
            early_stopping = EarlyStopping(patience=patience, min_delta = 0.001)

            training_log = {'val_loss': [], 'pred_loss': [], 'sigma': [], 'phi': [], 'tau': []}

            # Training/evaluation loop
            for epoch in range(100):
                print(epoch)
                # Train for one epoch
                model.train()
                if (epoch >= Update_init) & (epoch % Update_step == 0):
                    model.theta.requires_grad = True
                else:
                    model.theta.requires_grad = False
                for batch_idx, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    decorrelated_preds, decorrelated_targets, est = model(batch)
                    loss = torch.nn.functional.mse_loss(decorrelated_preds[:batch_size], decorrelated_targets[:batch_size])
                    metric = torch.nn.functional.mse_loss(est[:batch_size], batch.y[:batch_size])
                    loss.backward()
                    optimizer.step()
                # Compute predictions on held-out test test
                model.eval()
                _, _, val_est = model(data_val)
                val_loss = torch.nn.functional.mse_loss(val_est, data_val.y).item()
                lr_scheduler(val_loss)
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print('End at epoch' + str(epoch))
                    break
                _, _, pred_test = model(data_test)
                pred_loss = torch.nn.functional.mse_loss(pred_test, data_test.y).item()
                training_log["val_loss"].append(val_loss)
                training_log["pred_loss"].append(pred_loss)
                training_log["sigma"].append(model.theta[0].item())
                training_log["phi"].append(model.theta[1].item())
                training_log["tau"].append(model.theta[2].item())
            end_time = time.time()
            usertime = end_time - start_time
            t = np.append(t, usertime)
            size_vec = np.append(size_vec, n)
            epoch_vec = np.append(epoch_vec, epoch)

            df = pd.DataFrame({'time': t,
                       'epoch': epoch_vec,
                       'size': size_vec
                       })

            df.to_csv("/Users/zhanwentao/Documents/Abhi/NN/NN-GLS/running_time.csv")

if False:
    sampled_data = next(iter(train_loader))
    temp = [(idx, letter) for idx, letter in enumerate(train_loader)]

    batch = temp[1][1]

    mlp = torch.nn.Sequential(
        torch.nn.Linear(5, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1),
    )
    batch.o = mlp(batch.x)
    batch.size = batch_size
    edge_list = list()
    for i in range(batch.x.shape[0]):
        edge_list.append(batch.edge_attr[torch.where(batch.edge_index[1, :] == i)].squeeze())
    batch.edge_list = edge_list

    gather_neighbor_positions = GatherNeighborPositionsConv(20, 2)
    neighbor_positions = gather_neighbor_positions(
        batch.pos, batch.edge_index, batch.edge_attr, batch.batch_size
    )

    gather_neighbor_outputs1 = GatherNeighborInfoConv1(20)
    gather_neighbor_outputs = GatherNeighborInfoConv(20)
    neighbor_outputs = gather_neighbor_outputs(
        batch.o.double(), batch.edge_index, batch.edge_attr, batch.batch_size
    )

    compute_covariance_vectors = CovarianceVectorConv(nn, theta)

    compute_inverse_cov_matrices = InverseCovMatrixFromPositions(nn, 2, theta)
    Inv_Cov_Ni_Ni = compute_inverse_cov_matrices(neighbor_positions, batch.edge_list)
    Cov_i_Ni = compute_covariance_vectors(
        batch.pos, batch.edge_index, batch.edge_attr, batch.batch_size
    )

    B_i = torch.matmul(Inv_Cov_Ni_Ni, Cov_i_Ni.unsqueeze(2)).squeeze()
    Fi = theta[0] + theta[2] - torch.sum(B_i * Cov_i_Ni, dim=1)
    y_neighbor = gather_neighbor_outputs(
        batch.y, batch.edge_index, batch.edge_attr , batch.batch_size
    )
    y_decor = (batch.y - torch.sum(y_neighbor * B_i, dim=1)) / torch.sqrt(Fi)

    if False:
        ones = torch.ones(B_i.size(0), 1)
        negative_extended_b_vectors = torch.cat((ones, -B_i), dim=1)
        v_vectors = negative_extended_b_vectors / torch.sqrt(Fi.unsqueeze(1))
        neighbor_outputs = gather_neighbor_outputs1(batch.y, batch.edge_index, batch.edge_attr)
        decorrelated_pres_1 = (v_vectors * neighbor_outputs).sum(dim=1)

def predict(self, data_train, data_test, CI = False, **kwargs):
    with torch.no_grad():
        w_train = data_train.y - self.estimate(data_train.x)
        if CI:
            w_test, w_u, w_l = pyNNGLS.krig_pred(w_train, data_train.pos, data_test.pos, self.theta, **kwargs)
            estimation_test = self.estimate(data_test.x)
            return [estimation_test + w_test, estimation_test + w_u, estimation_test + w_l]
        else:
            w_test, _, _ = pyNNGLS.krig_pred(w_train, data_train.pos, data_test.pos, self.theta, **kwargs)
            estimation_test = self.estimate(data_test.x)
            return estimation_test + w_test