import torch
from .controldiffeq import natural_cubic_spline_coeffs
from utils.base_dataloader import load_base_data, load_auxiliary_data

class load_stgncde_base_data(load_base_data):
    def _get_augment_coeffs(self, data):
        train_x, _, val_x, _, test_x, _ = data

        times = torch.linspace(0, self.num_of_predict - 1, self.num_of_predict)

        augmented_train_x = []
        augmented_train_x.append(
            times.unsqueeze(0).unsqueeze(0).repeat(train_x.shape[0], train_x.shape[1], 1)
            .unsqueeze(-1).transpose(1,2))
        augmented_train_x.append(train_x[..., :].permute(0, 3, 1, 2))
        train_coeffs = torch.cat(augmented_train_x, dim=3)

        augmented_val_x = []
        augmented_val_x.append(
            times.unsqueeze(0).unsqueeze(0).repeat(val_x.shape[0], val_x.shape[1], 1)
            .unsqueeze(-1).transpose(1, 2))
        augmented_val_x.append(val_x[..., :].permute(0, 3, 1, 2))
        valid_coeffs = torch.cat(augmented_val_x, dim=3)

        augmented_test_x = []
        augmented_test_x.append(
            times.unsqueeze(0).unsqueeze(0).repeat(test_x.shape[0], test_x.shape[1], 1)
            .unsqueeze(-1).transpose(1,2))
        augmented_test_x.append(test_x[..., :].permute(0, 3, 1, 2))
        test_coeffs = torch.cat(augmented_test_x, dim=3)

        train_coeffs = natural_cubic_spline_coeffs(times, train_coeffs.transpose(1, 2))
        valid_coeffs = natural_cubic_spline_coeffs(times, valid_coeffs.transpose(1, 2))
        test_coeffs = natural_cubic_spline_coeffs(times, test_coeffs.transpose(1, 2))

        return (train_coeffs, valid_coeffs, test_coeffs)

    def load_data(self):
        raw_data = self._get_raw_data()
        data = self._split_data(raw_data)
        data, data_scalar = self._scale_data(data)
        timestamp, time_per_day = self._get_timestamp(raw_data)
        timestamp = self._split_timestamp(timestamp)
        coeffs = self._get_augment_coeffs(data)

        train_x, train_y, val_x, val_y, test_x, test_y = data
        if self.imputate_ratio > 0:
            train_x = self._imputate_data(train_x, self.imputate_ratio)
        train_ts, val_ts, test_ts = timestamp
        train_coeffs, val_coeffs, test_coeffs = coeffs

        train_dataset = torch.utils.data.TensorDataset(train_x, train_y, train_ts, *train_coeffs)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # ------- val_loader -------
        val_dataset = torch.utils.data.TensorDataset(val_x, val_y, val_ts, *val_coeffs)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        # ------- test_loader -------
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y, test_ts, *test_coeffs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print('train:', train_x.size(), train_y.size())
        print('val:', val_x.size(), val_y.size())
        print('test:', test_x.size(), test_y.size())

        return train_loader, val_loader, test_loader, data_scalar

    def get_train_data(self, batch):
        _, _, _, *coeffs = batch
        coeffs = [i.to(self.device) for i in coeffs]
        return coeffs

    def get_finetune_data(self, batch):
        _, _, _, *coeffs = batch
        coeffs = [i.to(self.device) for i in coeffs]
        return coeffs

    def get_adjust_data(self, batch):
        x, _, t, *_ = batch
        return x.to(self.device), t.to(self.device)

    def get_ground_truth(self, batch):
        _, y, _, *_ = batch
        return y.to(self.device)


class load_stgncde_auxiliary_data(load_auxiliary_data):
    def _get_augment_coeffs(self, data):
        train_x, _, val_x, _, test_x, _ = data

        times = torch.linspace(0, self.num_of_predict - 1, self.num_of_predict)

        augmented_train_x = []
        augmented_train_x.append(
            times.unsqueeze(0).unsqueeze(0).repeat(train_x.shape[0], train_x.shape[1], 1)
            .unsqueeze(-1).transpose(1,2))
        augmented_train_x.append(train_x[..., :].permute(0, 3, 1, 2))
        train_coeffs = torch.cat(augmented_train_x, dim=3)

        augmented_val_x = []
        augmented_val_x.append(
            times.unsqueeze(0).unsqueeze(0).repeat(val_x.shape[0], val_x.shape[1], 1)
            .unsqueeze(-1).transpose(1, 2))
        augmented_val_x.append(val_x[..., :].permute(0, 3, 1, 2))
        valid_coeffs = torch.cat(augmented_val_x, dim=3)

        augmented_test_x = []
        augmented_test_x.append(
            times.unsqueeze(0).unsqueeze(0).repeat(test_x.shape[0], test_x.shape[1], 1)
            .unsqueeze(-1).transpose(1,2))
        augmented_test_x.append(test_x[..., :].permute(0, 3, 1, 2))
        test_coeffs = torch.cat(augmented_test_x, dim=3)

        train_coeffs = natural_cubic_spline_coeffs(times, train_coeffs.transpose(1, 2))
        valid_coeffs = natural_cubic_spline_coeffs(times, valid_coeffs.transpose(1, 2))
        test_coeffs = natural_cubic_spline_coeffs(times, test_coeffs.transpose(1, 2))

        return (train_coeffs, valid_coeffs, test_coeffs)

    def load_data(self):
        raw_data = self._get_raw_data()
        data = self._split_data(raw_data)
        data, data_scalar = self._scale_data(data)
        timestamp, time_per_day = self._get_timestamp(raw_data)
        timestamp = self._split_timestamp(timestamp)
        tid = self._get_timeinday(raw_data)
        coeffs = self._get_augment_coeffs(data)

        train_x, train_y, val_x, val_y, test_x, test_y = data
        if self.imputate_ratio > 0:
            train_x = self._imputate_data(train_x, self.imputate_ratio)
        train_ts, val_ts, test_ts = timestamp
        train_tid, val_tid, test_tid = tid
        train_coeffs, val_coeffs, test_coeffs = coeffs

        train_dataset = torch.utils.data.TensorDataset(train_x, train_y, train_ts, train_tid, *train_coeffs)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # ------- val_loader -------
        val_dataset = torch.utils.data.TensorDataset(val_x, val_y, val_ts, val_tid, *val_coeffs)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        # ------- test_loader -------
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y, test_ts, test_tid, *test_coeffs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print('train:', train_x.size(), train_y.size())
        print('val:', val_x.size(), val_y.size())
        print('test:', test_x.size(), test_y.size())

        return train_loader, val_loader, test_loader, data_scalar

    def get_train_data(self, batch):
        _, _, _, _, *coeffs = batch
        coeffs = [i.to(self.device) for i in coeffs]
        return coeffs

    def get_finetune_data(self, batch):
        _, _, _, _, *coeffs = batch
        coeffs = [i.to(self.device) for i in coeffs]
        return coeffs

    def get_adjust_data(self, batch):
        x, _, ts, tid, *_ = batch
        return x.to(self.device), ts.to(self.device), tid.to(self.device)

    def get_ground_truth(self, batch):
        _, y, _, _, *_ = batch
        return y.to(self.device)
