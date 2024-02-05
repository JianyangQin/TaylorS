import torch
from utils.base_dataloader import load_base_data, load_auxiliary_data
from utils.timefeatures import get_timestamp, convert_array2timestamp
from torch.utils.data.distributed import DistributedSampler

class load_d2stgnn_base_data(load_base_data):
    def _get_coeffs(self, data):
        if self.dataset in ['HZ-METRO', 'SH-METRO']:
            train, val, test = data[0], data[1], data[2]
            train_ts, val_ts, test_ts = train['xtime'], val['xtime'], test['xtime']

            train_ts = convert_array2timestamp(train_ts, 'discrete', 2, self.freq)
            val_ts = convert_array2timestamp(val_ts, 'discrete', 2, self.freq)
            test_ts = convert_array2timestamp(test_ts, 'discrete', 2, self.freq)

            time_per_day = self.time_per_day

        else:
            len_data = len(data)
            timestamp, time_per_day = get_timestamp(embed='discrete',
                                                    features=2,
                                                    start_time=self.start_time,
                                                    num_of_steps=len_data,
                                                    freq=self.freq,
                                                    num_of_history=self.num_of_history,
                                                    num_of_predict=self.num_of_predict,
                                                    train_ratio=self.train_ratio,
                                                    val_ratio=self.val_ratio,
                                                    test_ratio=self.test_ratio)

            timestamp_x, _ = self._timestamp_seq2instance(timestamp, self.num_of_history, self.num_of_predict)

            len_total = len(timestamp_x)
            len_train = round(len_total * self.train_ratio)
            len_test = round(len_total * self.test_ratio)
            len_val = len_total - len_train - len_test

            train_ts = timestamp_x[:len_train]
            val_ts = timestamp_x[len_train:len_train + len_val]
            test_ts = timestamp_x[-len_test:]

        train_ts = torch.from_numpy(train_ts).type(torch.FloatTensor)
        val_ts = torch.from_numpy(val_ts).type(torch.FloatTensor)
        test_ts = torch.from_numpy(test_ts).type(torch.FloatTensor)

        # get days_in_week & times_in_day
        train_diw, train_tid = train_ts[..., 0:1], train_ts[..., 1:2]
        val_diw, val_tid = val_ts[..., 0:1], val_ts[..., 1:2]
        test_diw, test_tid = test_ts[..., 0:1], test_ts[..., 1:2]

        # nomalize & reshape
        train_tid /= time_per_day
        val_tid /= time_per_day
        test_tid /= time_per_day

        train_diw = train_diw.unsqueeze(1).repeat(1, self.num_of_vertices, 1, 1)
        train_tid = train_tid.unsqueeze(1).repeat(1, self.num_of_vertices, 1, 1)
        val_diw = val_diw.unsqueeze(1).repeat(1, self.num_of_vertices, 1, 1)
        val_tid = val_tid.unsqueeze(1).repeat(1, self.num_of_vertices, 1, 1)
        test_diw = test_diw.unsqueeze(1).repeat(1, self.num_of_vertices, 1, 1)
        test_tid = test_tid.unsqueeze(1).repeat(1, self.num_of_vertices, 1, 1)

        train_diw, train_tid = train_diw.permute(0, 1, 3, 2), train_tid.permute(0, 1, 3, 2)
        val_diw, val_tid = val_diw.permute(0, 1, 3, 2), val_tid.permute(0, 1, 3, 2)
        test_diw, test_tid = test_diw.permute(0, 1, 3, 2), test_tid.permute(0, 1, 3, 2)

        # concatanate (batch, seq, nodes, feats)
        train_coeffs = torch.cat([train_tid, train_diw], dim=2)
        val_coeffs = torch.cat([val_tid, val_diw], dim=2)
        test_coeffs = torch.cat([test_tid, test_diw], dim=2)

        return (train_coeffs, val_coeffs, test_coeffs)

    def load_data(self):
        raw_data = self._get_raw_data()
        data = self._split_data(raw_data)
        data, data_scalar = self._scale_data(data)
        timestamp, time_per_day = self._get_timestamp(raw_data)
        timestamp = self._split_timestamp(timestamp)
        coeffs = self._get_coeffs(raw_data)

        train_x, train_y, val_x, val_y, test_x, test_y = data
        if self.imputate_ratio > 0:
            train_x = self._imputate_data(train_x, self.imputate_ratio)
        train_ts, val_ts, test_ts = timestamp
        train_coeffs, val_coeffs, test_coeffs = coeffs

        train_dataset = torch.utils.data.TensorDataset(train_x, train_y, train_ts, train_coeffs)
        # if self.use_multi_gpu:
        #     train_dataset = DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # ------- val_loader -------
        val_dataset = torch.utils.data.TensorDataset(val_x, val_y, val_ts, val_coeffs)
        # if self.use_multi_gpu:
        #     val_dataset = DistributedSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        # ------- test_loader -------
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y, test_ts, test_coeffs)
        # if self.use_multi_gpu:
        #     test_dataset = DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print('train:', train_x.size(), train_y.size())
        print('val:', val_x.size(), val_y.size())
        print('test:', test_x.size(), test_y.size())

        return train_loader, val_loader, test_loader, data_scalar

    def get_train_data(self, batch):
        x, _, _, coeffs = batch
        return torch.cat([x, coeffs], dim=2).to(self.device)

    def get_finetune_data(self, batch):
        x, _, _, coeffs = batch
        return torch.cat([x, coeffs], dim=2).to(self.device)

    def get_adjust_data(self, batch):
        x, _, t, _ = batch
        return x.to(self.device), t.to(self.device)

    def get_ground_truth(self, batch):
        _, y, _, _ = batch
        return y.to(self.device)

class load_d2stgnn_auxiliary_data(load_auxiliary_data):
    def _get_coeffs(self, data):
        if self.dataset in ['HZ-METRO', 'SH-METRO']:
            train, val, test = data[0], data[1], data[2]
            train_ts, val_ts, test_ts = train['xtime'], val['xtime'], test['xtime']

            train_ts = convert_array2timestamp(train_ts, 'discrete', 2, self.freq)
            val_ts = convert_array2timestamp(val_ts, 'discrete', 2, self.freq)
            test_ts = convert_array2timestamp(test_ts, 'discrete', 2, self.freq)

            time_per_day = self.time_per_day
        elif self.dataset in ['Air-Quality']:
            len_train, len_val, len_test = len(data[0]['x']), len(data[1]['x']), len(data[2]['x'])
            len_data = len_train + len_val + len_test + 47
            timestamp, time_per_day = get_timestamp(
                embed=self.time_embed,
                features=self.time_features,
                start_time=self.start_time,
                num_of_steps=len_data,
                freq=self.freq,
                num_of_history=self.num_of_history,
                num_of_predict=self.num_of_predict,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio
            )

            timestamp_x, _ = self._timestamp_seq2instance(timestamp, self.num_of_history, self.num_of_predict)

            len_total = len(timestamp_x)
            len_train = round(len_total * self.train_ratio)
            len_test = round(len_total * self.test_ratio)
            len_val = len_total - len_train - len_test

            train_ts = timestamp_x[:len_train]
            val_ts = timestamp_x[len_train:len_train + len_val]
            test_ts = timestamp_x[-len_test:]

        else:
            len_data = len(data)
            timestamp, time_per_day = get_timestamp(embed='discrete',
                                                    features=2,
                                                    start_time=self.start_time,
                                                    num_of_steps=len_data,
                                                    freq=self.freq,
                                                    num_of_history=self.num_of_history,
                                                    num_of_predict=self.num_of_predict,
                                                    train_ratio=self.train_ratio,
                                                    val_ratio=self.val_ratio,
                                                    test_ratio=self.test_ratio)

            timestamp_x, _ = self._timestamp_seq2instance(timestamp, self.num_of_history, self.num_of_predict)

            len_total = len(timestamp_x)
            len_train = round(len_total * self.train_ratio)
            len_test = round(len_total * self.test_ratio)
            len_val = len_total - len_train - len_test

            train_ts = timestamp_x[:len_train]
            val_ts = timestamp_x[len_train:len_train + len_val]
            test_ts = timestamp_x[-len_test:]

        train_ts = torch.from_numpy(train_ts).type(torch.FloatTensor)
        val_ts = torch.from_numpy(val_ts).type(torch.FloatTensor)
        test_ts = torch.from_numpy(test_ts).type(torch.FloatTensor)

        # get days_in_week & times_in_day
        train_diw, train_tid = train_ts[..., 0:1], train_ts[..., 1:2]
        val_diw, val_tid = val_ts[..., 0:1], val_ts[..., 1:2]
        test_diw, test_tid = test_ts[..., 0:1], test_ts[..., 1:2]

        # nomalize & reshape
        train_tid /= time_per_day
        val_tid /= time_per_day
        test_tid /= time_per_day

        train_diw = train_diw.unsqueeze(1).repeat(1, self.num_of_vertices, 1, 1)
        train_tid = train_tid.unsqueeze(1).repeat(1, self.num_of_vertices, 1, 1)
        val_diw = val_diw.unsqueeze(1).repeat(1, self.num_of_vertices, 1, 1)
        val_tid = val_tid.unsqueeze(1).repeat(1, self.num_of_vertices, 1, 1)
        test_diw = test_diw.unsqueeze(1).repeat(1, self.num_of_vertices, 1, 1)
        test_tid = test_tid.unsqueeze(1).repeat(1, self.num_of_vertices, 1, 1)

        train_diw, train_tid = train_diw.permute(0, 1, 3, 2), train_tid.permute(0, 1, 3, 2)
        val_diw, val_tid = val_diw.permute(0, 1, 3, 2), val_tid.permute(0, 1, 3, 2)
        test_diw, test_tid = test_diw.permute(0, 1, 3, 2), test_tid.permute(0, 1, 3, 2)

        # concatanate (batch, seq, nodes, feats)
        train_coeffs = torch.cat([train_tid, train_diw], dim=2)
        val_coeffs = torch.cat([val_tid, val_diw], dim=2)
        test_coeffs = torch.cat([test_tid, test_diw], dim=2)

        return (train_coeffs, val_coeffs, test_coeffs)

    def load_data(self):
        raw_data = self._get_raw_data()
        data = self._split_data(raw_data)
        data, data_scalar = self._scale_data(data)
        timestamp, time_per_day = self._get_timestamp(raw_data)
        timestamp = self._split_timestamp(timestamp)
        tid = self._get_timeinday(raw_data)
        coeffs = self._get_coeffs(raw_data)

        train_x, train_y, val_x, val_y, test_x, test_y = data
        if self.imputate_ratio > 0:
            train_x = self._imputate_data(train_x, self.imputate_ratio)
        train_ts, val_ts, test_ts = timestamp
        train_tid, val_tid, test_tid = tid
        train_coeffs, val_coeffs, test_coeffs = coeffs

        train_dataset = torch.utils.data.TensorDataset(train_x, train_y, train_ts, train_tid, train_coeffs)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # ------- val_loader -------
        val_dataset = torch.utils.data.TensorDataset(val_x, val_y, val_ts, val_tid, val_coeffs)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        # ------- test_loader -------
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y, test_ts, test_tid, test_coeffs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print('train:', train_x.size(), train_y.size())
        print('val:', val_x.size(), val_y.size())
        print('test:', test_x.size(), test_y.size())

        return train_loader, val_loader, test_loader, data_scalar

    def get_train_data(self, batch):
        x, _, _, _, coeffs = batch
        return torch.cat([x, coeffs], dim=2).to(self.device)

    def get_finetune_data(self, batch):
        x, _, _, _, coeffs = batch
        return torch.cat([x, coeffs], dim=2).to(self.device)

    def get_adjust_data(self, batch):
        x, _, t, tid, _ = batch
        return x.to(self.device), t.to(self.device), tid.to(self.device)

    def get_ground_truth(self, batch):
        _, y, _, _, _ = batch
        return y.to(self.device)