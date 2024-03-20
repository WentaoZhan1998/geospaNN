import torch
from torch_geometric.nn import MessagePassing
from .utils import make_cov_full, distance, edit_batch, krig_pred

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

__all__ = ['nngls']