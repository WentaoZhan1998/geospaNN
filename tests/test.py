import torch
import nngls
import numpy as np
import time

def f5(X): return (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] +5*X[:,4])/6

sigma = 1
phi = 3
tau = 0.01
theta = [sigma, phi / np.sqrt(2), tau]

p = 5; funXY = f5

n = 1000
nn = 20
batch_size = 50

X, Y, coord, cov, corerr = Simulation(n, p, nn, funXY, theta, range = [0,1])

data = make_graph(X, Y, coord, nn)

train_loader, data_train, data_test = split_data(X, Y, coord, batch_size = batch_size, neighbor_size = 20,
                                                 test_proportion = 0.2)

start_time = time.time()
mlp = torch.nn.Sequential(
            torch.nn.Linear(5, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1),
        )
model = nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp = mlp, theta = torch.tensor(theta))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

training_log = {'loss': [], 'metric': [], 'sigma': [], 'phi': [], 'tau': []}

# Training/evaluation loop
for epoch in range(100):
    # Train for one epoch
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        print(batch_idx)
        optimizer.zero_grad()
        decorrelated_preds, decorrelated_targets, preds = model(batch)
        loss = torch.nn.functional.mse_loss(decorrelated_preds[:batch_size], decorrelated_targets[:batch_size])
        metric = torch.nn.functional.mse_loss(preds[:batch_size], batch.y[:batch_size])
        loss.backward()
        optimizer.step()
    # Compute predictions on held-out test test
    model.eval()
    decorrelated_preds, decorrelated_targets, preds = model(data_test)
    training_log["loss"].append(torch.nn.functional.mse_loss(decorrelated_preds, decorrelated_targets).item())
    training_log["metric"].append(torch.nn.functional.mse_loss(preds, data_test.y).item())
    training_log["sigma"].append(model.theta[0])
    training_log["phi"].append(model.theta[1])
    training_log["tau"].append(model.theta[2])
end_time = time.time()
usertime2 = end_time - start_time


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
        batch.pos, batch.edge_index, batch.edge_attr
    )

    gather_neighbor_outputs1 = GatherNeighborInfoConv1(20)
    gather_neighbor_outputs = GatherNeighborInfoConv(20)
    neighbor_outputs = gather_neighbor_outputs(batch.o.double(), batch.edge_index, batch.edge_attr)

    compute_covariance_vectors = CovarianceVectorConv(nn, theta)
    compute_covariance_vectors(batch.pos, batch.edge_index, batch.edge_attr)

    compute_inverse_cov_matrices = InverseCovMatrixFromPositions(nn, 2, theta)
    Inv_Cov_Ni_Ni = compute_inverse_cov_matrices(neighbor_positions, batch.edge_list)
    Cov_i_Ni = compute_covariance_vectors(batch.pos, batch.edge_index, batch.edge_attr)

    B_i = torch.matmul(Inv_Cov_Ni_Ni, Cov_i_Ni.unsqueeze(2)).squeeze()
    F_i = theta[0] + theta[2] - torch.sum(B_i * Cov_i_Ni, dim=1)
    y_neighbor = gather_neighbor_outputs(batch.y, batch.edge_index, batch.edge_attr)
    y_decor = (batch.y - torch.sum(y_neighbor * B_i, dim=1)) / torch.sqrt(F_i)

    if False:
        ones = torch.ones(B_i.size(0), 1)
        negative_extended_b_vectors = torch.cat((ones, -B_i), dim=1)
        v_vectors = negative_extended_b_vectors / torch.sqrt(F_i.unsqueeze(1))
        neighbor_outputs = gather_neighbor_outputs1(batch.y, batch.edge_index, batch.edge_attr)
        decorrelated_preds_1 = (v_vectors * neighbor_outputs).sum(dim=1)

