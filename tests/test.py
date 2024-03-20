import torch
#import nngls
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
X, Y, coord, cov, corerr = Simulation(n, p, nn, funXY, theta, range=[0, 1])
data = make_graph(X, Y, coord, nn)
data_train, data_val, data_test = split_data(X, Y, coord, neighbor_size=20,
                                             test_proportion=0.2)

mlp = torch.nn.Sequential(
    torch.nn.Linear(p, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 1)
)
nn_model = nn_train(mlp, lr =  0.01, min_delta = 0.001)
training_log = nn_model.train(data_train, data_val, data_test)
theta_update(torch.tensor([1, 1.5, 0.01]), mlp(data_train.x).squeeze() - data_train.y, data_train.pos, neighbor_size = 20)
mlp = torch.nn.Sequential(
    torch.nn.Linear(p, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 1)
)
model = nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp, theta=torch.tensor([1.5, 5, 0.1]))
nngls_model = nngls_train(model, lr =  0.01, min_delta = 0.001)
training_log = nngls_model.train(data_train, data_val, data_test,
                                 Update_init = 10, Update_step = 10)

training_log = {'val_loss': [], 'pred_loss': [], 'sigma': [], 'phi': [], 'tau': []}

if False:
    residual = data_train.y - model(data_train)[2]
    residual = residual.detach().numpy()
    coord = data_train.pos.detach().numpy()
    dist = distance_np(coord, coord)
    rank = make_rank(data_train.pos, model.neighbor_size)


    def test2(theta):
        sigma, phi, tau = theta
        cov = sigma * (np.exp(-phi * dist) + tau * np.eye(600))  # need dist, n

        term1 = 0
        term2 = 0
        for i in range(600):
            ind = rank[i, :][rank[i, :] <= i]
            id = np.append(ind, i)

            sub_cov = cov[ind, :][:, ind]
            if np.linalg.det(sub_cov):
                bi = np.linalg.solve(cov[ind, :][:, ind], cov[ind, i])
            else:
                bi = np.zeros(ind.shape)
            I_B_i = np.append(-bi, 1)
            F_i = cov[i, i] - np.inner(cov[ind, i], bi)
            decor_res = np.sqrt(np.reciprocal(F_i)) * np.dot(I_B_i, residual[id])
            term1 += np.log(F_i)
            term2 += decor_res ** 2
        return (term1 + term2)


    def constraint1(x):
        return x[0]


    def constraint2(x):
        return x[1]


    def constraint3(x):
        return x[2]


    cons = [{'type': 'ineq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint3}]

    res = minimize(test2, np.array([1, 3, 0.01]), constraints=cons)
    # sigma_temp, phi_temp, tau_temp = theta_hat.detach().numpy()
    # print('det')
    # print(np.linalg.det(sigma_temp*(np.exp(-phi_temp * dist) + tau_temp * np.eye(n))))
    theta_hat_new = res.x

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
    F_i = theta[0] + theta[2] - torch.sum(B_i * Cov_i_Ni, dim=1)
    y_neighbor = gather_neighbor_outputs(
        batch.y, batch.edge_index, batch.edge_attr , batch.batch_size
    )
    y_decor = (batch.y - torch.sum(y_neighbor * B_i, dim=1)) / torch.sqrt(F_i)

    if False:
        ones = torch.ones(B_i.size(0), 1)
        negative_extended_b_vectors = torch.cat((ones, -B_i), dim=1)
        v_vectors = negative_extended_b_vectors / torch.sqrt(F_i.unsqueeze(1))
        neighbor_outputs = gather_neighbor_outputs1(batch.y, batch.edge_index, batch.edge_attr)
        decorrelated_pres_1 = (v_vectors * neighbor_outputs).sum(dim=1)

