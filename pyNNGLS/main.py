from .utils import LRScheduler, EarlyStopping, split_loader, theta_update

import torch
from typing import Optional

class nn_train():
    def __init__(
            self,
            model,
            lr: Optional[float] =  0.01,
            patience: Optional[int] = 10,
            min_delta=0.001
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = LRScheduler(self.optimizer, patience=int(patience/2), factor=0.5)
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    def train(self,
              data_train,
              data_val,
              data_test: Optional = None,
              batch_size: Optional[int] = None,
              epoch_num: Optional[int] = 100,
              seed: Optional[int] = torch.randint(0, 2024, (1,))
              ):
        if batch_size is None:
            batch_size = int(data_train.x.shape[0]/10)
        torch.manual_seed(seed)
        train_loader = split_loader(data_train, batch_size)
        training_log = {'val_loss': [], 'pred_loss': [], 'sigma': [], 'phi': [], 'tau': []}
        for epoch in range(epoch_num):
            # Train for one epoch
            #print(epoch)
            self.model.train()

            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                est = self.model(batch.x).squeeze()
                loss = torch.nn.functional.mse_loss(est, batch.y)
                loss.backward()
                self.optimizer.step()
            # Compute predictions on held-out test test
            self.model.eval()
            val_est = self.model(data_val.x).squeeze()
            val_loss = torch.nn.functional.mse_loss(val_est, data_val.y).item()
            self.lr_scheduler(val_loss)
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print('End at epoch' + str(epoch))
                break
            training_log["val_loss"].append(val_loss)
            if data_test is None:
                test_est = self.model(data_test.x).squeeze()
                pred_loss = torch.nn.functional.mse_loss(test_est, data_test.y).item()
                training_log["pred_loss"].append(pred_loss)

        return training_log

class nngls_train():
    def __init__(
            self,
            model,
            lr: Optional[float] =  0.01,
            patience: Optional[int] = 10,
            min_delta = 0.001
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = LRScheduler(self.optimizer, patience=int(patience/2), factor=0.5)
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    def theta_update(self, w, data):
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
              data_train,
              data_val,
              data_test: Optional = None,
              batch_size: Optional[int] = None,
              epoch_num: Optional[int] = 100,
              Update_init: Optional[int] = 0,
              Update_step: Optional[int] = 1,
              Update_bound: Optional[float] = 0.1,
              seed: Optional[int] = torch.randint(0, 2024, (1,))
              ):
        if batch_size is None:
            batch_size = int(data_train.x.shape[0]/10)
        torch.manual_seed(seed)
        train_loader = split_loader(data_train, batch_size)
        training_log = {'val_loss': [], 'pred_loss': [], 'sigma': [], 'phi': [], 'tau': []}
        for epoch in range(epoch_num):
            # Train for one epoch
            w = data_train.y - self.model.estimate(data_train.x)
            self.model.train()
            self.model.theta.requires_grad = False
            if (epoch >= Update_init) & (epoch % Update_step == 0):
                self.theta_update(w, data_train)

            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                decorrelated_preds, decorrelated_targets, est = self.model(batch)
                loss = torch.nn.functional.mse_loss(decorrelated_preds[:batch_size], decorrelated_targets[:batch_size])
                loss.backward()
                self.optimizer.step()
            # Compute predictions on held-out test test
            self.model.eval()
            _, _, val_est = self.model(data_val)
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
            if data_test is None:
                _, _, test_est = self.model(data_test)
                pred_loss = torch.nn.functional.mse_loss(test_est, data_test.y).item()
                training_log["pred_loss"].append(pred_loss)

        return training_log

__all__ = ['nngls_train', 'nn_train']