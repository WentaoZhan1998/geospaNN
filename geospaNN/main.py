from .utils import LRScheduler, EarlyStopping, split_loader, theta_update

import torch
import torch_geometric
from typing import Optional

class nn_train():
    """
    A wrapper for training the ordinary neural networks (simple MLP).

    The class wraps up a standard training process for ordinary neural networks. Currently it only works for simple MLPs
    and will be extended to more complicated settings in the future. For more advanced model, users are recommended to write
    the training functions manually.

    Attributes:
        model (torch.nn.Module):
            A trainable feed-forward model that returns the output.
        lr (float):
            Learning rate.
        patience (float):
            The patience for the early stopping rule, see train() for more details.
        patience_cut_lr (float):
            The patience for cutting the learning rate, see train() for more details.
        min_delta (float):
            The threshold for terminating the training, see train() for more details.

    Methods:
        train():
            Train the model under a mean-squared loss and the early-stopping rule as follows.
            If the validation loss does not have a drop greater than min_delta for #patience_cut_lr epoches,
            reduce the learning rate by 50%.
            If the validation loss does not have a drop greater than min_delta for #patience epoches,
            the training process terminates.
            Since Adam optimizer is used here, cutting the learning rate is unnecessary, but we do find setting "patience_cut_lr =
            patience/2" helps the convergence in many scenarios. We keep this setting as default.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            lr: Optional[float] =  0.01,
            patience: Optional[int] = 10,
            patience_cut_lr: Optional[float] = None,
            min_delta: float = 0.001
    ):
        if patience_cut_lr is None:
            patience_cut_lr = int(patience/2)
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = LRScheduler(self.optimizer, patience=patience_cut_lr, factor=0.5)
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    def train(self,
              data_train: torch_geometric.data.Data,
              data_val: torch_geometric.data.Data,
              data_test: Optional[torch_geometric.data.Data] = None,
              batch_size: Optional[int] = None,
              epoch_num: Optional[int] = 100,
              seed: Optional[int] = torch.randint(0, 2024, (1,))
              ) -> list:
        """Train the neural networks model.

        Parameters:
            data_train:
                Training data containing x, y and spatial coordinates, can be the output of split_data() or make_graph().
            data_val:
                validation data containing x, y and spatial coordinates, can be the output of split_data() or make_graph().
            data_test:
                Testing data containing x, y and spatial coordinates, can be the output of split_data() or make_graph().
                If not specified, data_train is used for testing.
            batch_size:
                Individual size of mini-batches that data_train is split into.
            epoch_num:
                Maximum number of epoches allowed.
            seed:
                Random seed for data splitting.

        Returns:
            training_log:
                A list contains the validation loss, estimation loss.
        """
        if batch_size is None:
            batch_size = int(data_train.x.shape[0]/10)
        if data_test is None:
            data_test = data_train
        torch.manual_seed(seed)
        train_loader = split_loader(data_train, batch_size)
        training_log = {'val_loss': [], 'est_loss': [], 'sigma': [], 'phi': [], 'tau': []}
        for epoch in range(epoch_num):
            # Train for one epoch
            self.model.train()

            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                est = self.model(batch.x).squeeze()
                loss = torch.nn.functional.mse_loss(est, batch.y)
                loss.backward()
                self.optimizer.step()
            # Compute estimations on held-out test set
            self.model.eval()
            val_est = self.model(data_val.x).squeeze()
            val_loss = torch.nn.functional.mse_loss(val_est, data_val.y).item()
            self.lr_scheduler(val_loss)
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print('End at epoch' + str(epoch))
                break
            training_log["val_loss"].append(val_loss)
            test_est = self.model(data_test.x).squeeze()
            est_loss = torch.nn.functional.mse_loss(test_est, data_test.y).item()
            training_log["est_loss"].append(est_loss)

        return training_log

class nngls_train():
    """
    A wrapper for training the NN-GLS model.

    The class wraps up the training process for NN-GLS. We assume simple MLP is used for the upper body of the model.
    NN-GLS allows for more complicated network structures before the final decorrelation step.
    However, for more advanced structures, finer tuning on the hyperparameters is often needed.
    Users are recommended to write the training functions manually in that case.

    Attributes:
        model (torch.nn.Module):
            A trainable feed-forward model that returns the output.
        lr (float):
            Learning rate.
        patience (int):
            The patience for the early stopping rule, see train() for more details.
        patience_cut_lr (float):
            The patience for cutting the learning rate, see train() for more details.
        min_delta (float):
            The threshold for terminating the training, see train() for more details.

    Methods:
        train():
            Same as nn_train.train(), train the model under a mean-squared loss and the early-stopping rule as follows.
            If the validation loss does not have a drop greater than min_delta for #patience_cut_lr epoches,
            reduce the learning rate by 50%.
            If the validation loss does not have a drop greater than min_delta for #patience epoches,
            the training process terminates.
            Since Adam optimizer is used here, cutting the learning rate is unnecessary, but we do find setting "patience_cut_lr =
            patience/2" helps the convergence in many scenarios. We keep this setting as default.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            lr: Optional[float] =  0.01,
            patience: Optional[int] = 10,
            patience_cut_lr: Optional[int] = None,
            min_delta: float = 0.001
    ):
        if patience_cut_lr is None:
            patience_cut_lr = int(patience/2)
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = LRScheduler(self.optimizer, patience=patience_cut_lr, factor=0.5)
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    def theta_update(self,
                     w: torch.tensor,
                     data: torch_geometric.data.Data
                     ): #### Can be replaced by directly using theta_update?
        """Update the spatial parameters using maximum likelihood.

        This is a wrapper for theta_update() within the training module. See help(geospaNN.theta_update) for more details.

        Parameters:
            w:
                Length n observations of the spatial random effect without any fixed effect.
            data:
                The data.pos object should contain a nxd coordinates matrix.

        Returns:
            Update self.model.theta by the new estimation.
        """
        theta_new = theta_update(self.model.theta,
                                 w,
                                 data.pos,
                                 self.model.neighbor_size)

        state_dict = self.model.state_dict()
        state_dict['theta'] = torch.from_numpy(theta_new)
        self.model.load_state_dict(state_dict)
        print('to')
        print(theta_new)

    def train(self,
              data_train: torch_geometric.data.Data,
              data_val: torch_geometric.data.Data,
              data_test: Optional[torch_geometric.data.Data] = None,
              batch_size: Optional[int] = None,
              epoch_num: Optional[int] = 100,
              Update_init: Optional[int] = 0,
              Update_step: Optional[int] = 1,
              seed: Optional[int] = torch.randint(0, 2024, (1,)),
              vignette: Optional[bool] = False
              ) -> list:
        """Train NN-GLS.

        Parameters:
            data_train:
                Training data containing x, y and spatial coordinates, can be the output of split_data() or make_graph().
            data_val:
                validation data containing x, y and spatial coordinates, can be the output of split_data() or make_graph().
            data_test:
                Testing data containing x, y and spatial coordinates, can be the output of split_data() or make_graph().
                If not specified, data_train is used for testing.
            batch_size:
                Individual size of mini-batches that data_train is split into.
            epoch_num:
                Maximum number of epoches allowed.
            Update_init:
                Initial epoch to start spatial parameter updating. The aim here is to allow a 'burn-in' period for NN-GLS's
                fexed-effect estimation to converge. Default value is 0.
            Update_step:
                The spatial parameters will be updated every #Update_step epoches. The default value is 1.
            seed:
                Random seed for data splitting.

        Returns:
            training_log:
                A list contains the validation loss, estimation loss.
        """
        if batch_size is None:
            batch_size = int(data_train.x.shape[0]/10)
        torch.manual_seed(seed)
        train_loader = split_loader(data_train, batch_size)
        training_log = {'val_loss': [], 'est_loss': [], 'sigma': [], 'phi': [], 'tau': []}
        for epoch in range(epoch_num):
            # Train for one epoch
            w = data_train.y - self.model.estimate(data_train.x)
            self.model.train()
            self.model.theta.requires_grad = False
            if (epoch >= Update_init) & (epoch % Update_step == 0):
                self.theta_update(w, data_train)

            for batch_idx, batch in enumerate(train_loader):
                if vignette:
                    print(batch_idx)
                self.optimizer.zero_grad()
                decorrelated_preds, decorrelated_targets, est = self.model(batch)
                loss = torch.nn.functional.mse_loss(decorrelated_preds[:batch_size], decorrelated_targets[:batch_size])
                loss.backward()
                self.optimizer.step()
            # Compute estimations on held-out test set
            self.model.eval()
            val_est = self.model.estimate(data_val.x)
            val_loss = torch.nn.functional.mse_loss(val_est, data_val.y).item()
            self.lr_scheduler(val_loss)
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print('End at epoch' + str(epoch))
                break
            training_log["val_loss"].append(val_loss)
            training_log["sigma"].append(self.model.theta[0].item())
            training_log["phi"].append(self.model.theta[1].item())
            training_log["tau"].append(self.model.theta[2].item())
            if data_test is not None:
                test_est = self.model.estimate(data_test.x)
                est_loss = torch.nn.functional.mse_loss(test_est, data_test.y).item()
                training_log["est_loss"].append(est_loss)

        return training_log

__all__ = ['nngls_train', 'nn_train']