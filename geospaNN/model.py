from .utils import make_cov_full, distance, edit_batch, krig_pred

import torch
import torch_geometric
from torch_geometric.nn import MessagePassing
from typing import Callable, Optional, Tuple

class NeighborCovVec(MessagePassing):
    """
    A message-passing layer that returns covariance vectors for a single batch. For neighbor size k, and batch size b, the
    message-passing layer will return a bxp tensor, where the ith row is the covariance vector cov(i, N(i)).
    ...

    Attributes:
        neighbor_size (int):
            Size of nearest neighbor used. i.e. k in the documentation.

        theta (tuple[float, float, float]):
            theta[0], theta[1], theta[2] represent sigma^2, phi, tau in the exponential covariance family.

    Methods:
        forward():

        message():
    """
    def __init__(self, neighbor_size, theta):
        super().__init__(aggr="sum")
        self.neighbor_size = neighbor_size
        self.theta = theta

    def forward(self, pos, edge_index, edge_attr, batch_size):
        return self.propagate(edge_index, pos=pos, edge_attr=edge_attr)[range(batch_size), :]

    def message(self, pos_i, pos_j, edge_attr):
        num_edges = edge_attr.shape[0]
        msg = torch.zeros(num_edges, self.neighbor_size)
        col_idc = edge_attr.flatten().int()
        row_idc = torch.tensor(range(num_edges)).int()
        msg[row_idc, col_idc] = make_cov_full(distance(pos_i - pos_j, torch.zeros(1, 2)), self.theta).squeeze()
        return msg

class InverseCovMat(torch.nn.Module):
    """
    A feed-forward layer that returns inverses of nearest neighbor covariance matrices. For neighbor size k, and batch size b,
    the feed-forward layer will return a bxkxk tensor, where the ith kxk matrix is the inverse of neighbor covariance matrix
    cov(N(i), N(i)).
    ...

    Attributes:
        neighbor_size (int):
            Size of nearest neighbor used for NNGP approximation. i.e. k in the documentation.

        coord_dimension (int):
            Dimension of the coordinates, i.e. d in the documentation.

        theta (tuple[float, float, float]):
            theta[0], theta[1], theta[2] represent sigma^2, phi, tau in the exponential covariance family.

    Methods:
        forward():
    """
    def __init__(self, neighbor_size, coord_dimension, theta):
        super(InverseCovMat, self).__init__()
        self.neighbor_size = neighbor_size
        self.coord_dimension = coord_dimension
        self.theta = theta

    def forward(self, neighbor_positions, edge_list):
        batch_size = neighbor_positions.shape[0]
        neighbor_positions = neighbor_positions.reshape(-1, self.neighbor_size, self.coord_dimension)
        neighbor_positions1 = neighbor_positions.unsqueeze(1)
        neighbor_positions2 = neighbor_positions.unsqueeze(2)
        dists = torch.sqrt(torch.sum((neighbor_positions1 - neighbor_positions2) ** 2, axis=-1))
        cov = make_cov_full(dists, self.theta, nuggets=True)  # have to add nuggets (resolved)
        # cov_final = self.theta[0]*torch.eye(self.neighbor_size).repeat(batch_size, 1, 1)
        # for i in range(batch_size):
        #    cov_final[i, edge_list[i].reshape(1, -1, 1), edge_list[i].reshape(1, 1, -1)] = \
        #        cov[i, edge_list[i].reshape(1, -1, 1), edge_list[i].reshape(1, 1, -1)]
        # inv_cov_final = torch.linalg.inv(cov_final)
        inv_cov_final = torch.linalg.inv(cov)
        return inv_cov_final


class NeighborPositions(MessagePassing):
    """
    A message-passing layer that collect the coordinates of the nearest neighborhood. For neighbor size k, batch size b,
    and coordinates' dimension d, the message-passing layer will return a bx(k*d) tensor, where the ith row is the
    concatenation of k d-dimensional coordinates representing the k-nearest neighborhood of location i.
    ...

    Attributes:
        neighbor_size (int):
            Size of nearest neighbor used for NNGP approximation. i.e. k in the documentation.

        coord_dimension (int):
            Dimension of the coordinates, i.e. d in the documentation.

    Methods:
        forward():

        message():
    """
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
        row_idc = torch.tensor(range(num_edges)).int()
        msg[
            row_idc.unsqueeze(1), col_idc.unsqueeze(1) + torch.tensor(range(self.coord_dimensions))
        ] = pos_j
        return msg


class NeighborInfo(MessagePassing):
    """
    A message-passing layer that collect the output of the nearest neighborhood. For neighbor size k, batch size b,
    the message-passing layer will return a bxk tensor, where the ith row is the
    output from the size-p nearest neighborhood of location i.
    ...

    Attributes:
        neighbor_size (int):
            Size of nearest neighbor used for NNGP approximation. i.e. k in the documentation.

    Methods:
        forward():

        message():
    """
    def __init__(self, neighbor_size):
        super().__init__(aggr="sum")
        self.neighbor_size = neighbor_size

    def forward(self, y, edge_index, edge_attr, batch_size):
        out = self.propagate(edge_index, y=y.reshape(-1, 1), edge_attr=edge_attr)[range(batch_size), :]
        return out

    def message(self, y_j, edge_attr):
        num_edges = edge_attr.shape[0]
        msg = torch.zeros(num_edges, self.neighbor_size).double()
        col_idc = edge_attr.flatten().int()
        row_idc = torch.tensor(range(num_edges)).int()
        msg[row_idc, col_idc] = y_j.squeeze().double()
        return msg


class nngls(torch.nn.Module):
    """
    A feed-forward module implementing the NN-GLS algorithm from Zhan et.al 2023. Where the outputs and responses are
    spatially decorrelated using NNGP approximation proposed by Datta et.al 2016. The decorrelation is implemented by
    using the message passing (neighborhood aggregation) framework from troch_geometric package. 
    The aggregation only happens on the output layer, while the main body, i.e. the multi-layer perceptron, allows for
    flexible choice.
    ...

    Attributes:
        p (int):
            Number of features for prediction.
        neighbor_size (int):
            Size of nearest neighbor used for NNGP approximation. i.e. k in the documentation.
        coord_dimension (int):
            Dimension of the coordinates, i.e. d in the documentation.
        mlp (torch.nn.Module):
            Prespecified multi-layer perceptron that takes nxp covariates matrix as the input and nx1 vector as the output.
            Allows techniques like dropout.
        compute_covariance_vectors (MessagePassing):
            A message-passing layer returns the covariance vector between points and their neighbors. See NeighborCovVec()
            for more details.
        compute_inverse_cov_matrices (nn.module):
            A feed-forward layer computing the inverses of neighborhood in a vectorized form. See InverseCovMat()
            for more details.
        gather_neighbor_positions (MessagePassing):
            A message-passing layer that collects the positions of the neighbors in a compact form. See NeighborPositions()
            for more details.
        gather_neighbor_outputs (MessagePassing):
            Similar to gather_neighbor_positions, the function collects the scalar output (or other quantities) of the neighbors
            in a compact form. See NeighborInfo() for more details.

    Methods:
        forward():
            Take mini-batch as input and returns a tuple of the [decorrelated response, decorrelated output, original output].
            The outcomes are used in the training process defined in nngls_train().
        estimate():
            Return the estimation of the non-spatial effect with any covariates X. The input X must be of size nxp, where p
            is the number of features.
        predict():
            Apply kriging prediction on the testing dataset based on the estimated spatial effect on the training dataset.
    

    See Also:
    nngls_train : Training class for NN-GLS model. \

    Datta, Abhirup, et al. "Hierarchical nearest-neighbor Gaussian process models for large geostatistical datasets."
    Journal of the American Statistical Association 111.514 (2016): 800-812. \

    Datta, Abhirup. "Sparse nearest neighbor Cholesky matrices in spatial statistics."
    arXiv preprint arXiv:2102.13299 (2021). \

    ZZhan, Wentao, and Abhirup Datta. 2024. “Neural Networks for Geospatial Data.”
    Journal of the American Statistical Association, June, 1–21. doi:10.1080/01621459.2024.2356293.\
    """
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
        self.theta = torch.nn.Parameter(torch.Tensor(theta))  # split to accelerate?
        self.compute_covariance_vectors = NeighborCovVec(neighbor_size, self.theta)
        self.compute_inverse_cov_matrices = InverseCovMat(
            neighbor_size, coord_dimensions, self.theta
        )
        self.gather_neighbor_positions = NeighborPositions(neighbor_size, coord_dimensions)
        self.gather_neighbor_outputs = NeighborInfo(neighbor_size)

        # Simple MLP to map features to scalars
        self.mlp = mlp

    def forward(self, batch):
        """Feed-forward step with spatially decorrelated output.

        Parameters:
            batch: torch_geometric.data.Data
                A mini-batch of the data contains the x, y, coordinates, and the indexs of edges connecting the nearest neighbors.
                The mini-batch object can be created by the function split_loader().

        Returns:
            y_decor: torch.Tensor
                A decorrelated response vector computed from data.y.
            o_decor: torch.Tensor
                A decorrelated output vector computed from the output of the multi-layer perceptron self.mlp(batch.x).
            o: torch.Tensor
                The original output vector computed of the multi-layer perceptron.
        """
        if torch_geometric.__version__ >= '2.4.0':
            keys = batch.keys()
        else:
            keys = batch.keys
        if 'batch_size' not in keys: #### use batch.keys() in higher version of torch_geom
            batch.batch_size = batch.x.shape[0]  #### can improve
        batch = edit_batch(batch)
        Cov_i_Ni = self.compute_covariance_vectors(batch.pos, batch.edge_index, batch.edge_attr, batch.batch_size)
        coord_neighbor = self.gather_neighbor_positions(batch.pos, batch.edge_index, batch.edge_attr, batch.batch_size)
        Inv_Cov_Ni_Ni = self.compute_inverse_cov_matrices(coord_neighbor, batch.edge_list)

        B_i = torch.matmul(Inv_Cov_Ni_Ni, Cov_i_Ni.unsqueeze(2)).squeeze()
        F_i = self.theta[0] + self.theta[2] - torch.sum(B_i * Cov_i_Ni, dim=1)

        y_neighbor = self.gather_neighbor_outputs(batch.y, batch.edge_index, batch.edge_attr, batch.batch_size)
        y_decor = (batch.y[range(batch.batch_size)] - torch.sum(y_neighbor * B_i, dim=1)) / torch.sqrt(F_i)
        o = self.mlp(batch.x).squeeze().reshape(-1)
        o_neighbor = self.gather_neighbor_outputs(o, batch.edge_index, batch.edge_attr, batch.batch_size)
        o_decor = (o[range(batch.batch_size)] - torch.sum(o_neighbor * B_i, dim=1)) / torch.sqrt(F_i)

        return y_decor, o_decor, o

    def estimate(self, X: torch.Tensor
                 ) -> torch.Tensor:
        """Estimate the non-spatial effect with covariates X,

        Parameters:
            X:
                A nxp matrix where p is the number of features.

        Returns:
            estimation
        """
        assert X.shape[1] == self.p
        with torch.no_grad():
            return self.mlp(X).squeeze()

    def predict(self,
                data_train: torch_geometric.data.Data,
                data_test: torch_geometric.data.Data,
                CI:Optional[bool] = False, **kwargs
                ):
        """Kriging prediction on a test dataset.

        The function provides spatial prediction with the following steps.
        1: Apply the multi-layer perceptron on the training data for a fixed effect estimation.
        2: Compute the training residual as the estimated spatial effect (#### to implement: and estimate the spatial parameters).
        3: Use NNGP-approximated-kriging to predict the spatial effect and it's confidence interval.
        See krig_pred() for more details.
        4: Provide the overall prediction by combining the spatial effect prediction and fixed effect estimation.

        Parameters:
            data_train:
                Training data containing x, y and spatial coordinates, can be the output of split_data() or make_graph().
            data_test:
                Testing data containing x, y and spatial coordinates, can be the output of split_data() or make_graph().
            CI:
                A boolean value indicating whether to provide the 95% confidence intervals. (#### confidence level to add)

        Returns:
            if CI is True:
                tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                A tuple contains the prediction, confidence upper bound and confidence lower bound.
            else:
                torch.Tensor:
                only contains the prediction.

        See Also:
            krig_pred: Kriging prediction (Gaussian process regression) with confidence interval. \

            Zhan, Wentao, and Abhirup Datta. 2024. “Neural Networks for Geospatial Data.”
            Journal of the American Statistical Association, June, 1–21. doi:10.1080/01621459.2024.2356293.
        """
        with torch.no_grad():
            w_train = data_train.y - self.estimate(data_train.x)
            if CI:
                w_test, w_u, w_l = krig_pred(w_train, data_train.pos, data_test.pos, self.theta, **kwargs)
                estimation_test = self.estimate(data_test.x)
                return [estimation_test + w_test, estimation_test + w_u, estimation_test + w_l]
            else:
                w_test, _, _ = krig_pred(w_train, data_train.pos, data_test.pos, self.theta, **kwargs)
                estimation_test = self.estimate(data_test.x)
                return estimation_test + w_test


__all__ = ['nngls']
