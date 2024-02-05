import numpy as np
import torch
from utils.base_dataloader import load_base_data, load_auxiliary_data
from utils.timefeatures import get_timestamp, convert_array2timestamp

class load_gwnet_base_data(load_base_data):
    def _get_timeinday(self, data):
        if self.dataset in ['HZ-METRO', 'SH-METRO']:
            train, val, test = data[0], data[1], data[2]
            train_ts, val_ts, test_ts = train['xtime'], val['xtime'], test['xtime']

            train_ts = convert_array2timestamp(train_ts, 'discrete', 2, self.freq)
            val_ts = convert_array2timestamp(val_ts, 'discrete', 2, self.freq)
            test_ts = convert_array2timestamp(test_ts, 'discrete', 2, self.freq)

            train_tid = train_ts[..., 1:2] / self.time_per_day
            val_tid = val_ts[..., 1:2] / self.time_per_day
            test_tid = test_ts[..., 1:2] / self.time_per_day

            train_tid = np.repeat(np.expand_dims(train_tid, 1), self.num_of_vertices, axis=1).transpose((0, 1, 3, 2))
            val_tid = np.repeat(np.expand_dims(val_tid, 1), self.num_of_vertices, axis=1).transpose((0, 1, 3, 2))
            test_tid = np.repeat(np.expand_dims(test_tid, 1), self.num_of_vertices, axis=1).transpose((0, 1, 3, 2))

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

            tid = timestamp[..., 1:2] / self.time_per_day

            tid, _ = self._timestamp_seq2instance(tid, self.num_of_history, self.num_of_predict)
            tid = np.repeat(np.expand_dims(tid, 1), self.num_of_vertices, axis=1).transpose((0, 1, 3, 2))


            len_total = len(tid)
            len_train = round(len_total * self.train_ratio)
            len_test = round(len_total * self.test_ratio)
            len_val = len_total - len_train - len_test

            train_tid = tid[:len_train]
            val_tid = tid[len_train:len_train + len_val]
            test_tid = tid[-len_test:]

        train_tid = torch.from_numpy(train_tid).type(torch.FloatTensor)
        val_tid = torch.from_numpy(val_tid).type(torch.FloatTensor)
        test_tid = torch.from_numpy(test_tid).type(torch.FloatTensor)

        return (train_tid, val_tid, test_tid)

    def load_data(self):
        raw_data = self._get_raw_data()
        data = self._split_data(raw_data)
        data, data_scalar = self._scale_data(data)
        timestamp, time_per_day = self._get_timestamp(raw_data)
        timestamp = self._split_timestamp(timestamp)
        tid = self._get_timeinday(raw_data)

        train_x, train_y, val_x, val_y, test_x, test_y = data
        if self.imputate_ratio > 0:
            train_x = self._imputate_data(train_x, self.imputate_ratio)
        train_ts, val_ts, test_ts = timestamp
        train_tid, val_tid, test_tid = tid

        train_dataset = torch.utils.data.TensorDataset(train_x, train_y, train_ts, train_tid)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # ------- val_loader -------
        val_dataset = torch.utils.data.TensorDataset(val_x, val_y, val_ts, val_tid)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        # ------- test_loader -------
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y, test_ts, test_tid)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print('train:', train_x.size(), train_y.size())
        print('val:', val_x.size(), val_y.size())
        print('test:', test_x.size(), test_y.size())

        return train_loader, val_loader, test_loader, data_scalar

    def get_train_data(self, batch):
        x, _, _, tid = batch
        return torch.cat([x, tid], dim=2).to(self.device)

    def get_finetune_data(self, batch):
        x, _, _, tid = batch
        return torch.cat([x, tid], dim=2).to(self.device)

    def get_adjust_data(self, batch):
        x, _, t, _ = batch
        return x.to(self.device), t.to(self.device)

    def get_ground_truth(self, batch):
        _, y, _, _ = batch
        return y.to(self.device)


class load_gwnet_auxiliary_data(load_auxiliary_data):
    def get_train_data(self, batch):
        x, _, _, tid = batch
        return torch.cat([x, tid[..., :self.num_of_history]], dim=2).to(self.device)

    def get_finetune_data(self, batch):
        x, _, _, tid = batch
        return torch.cat([x, tid[..., :self.num_of_history]], dim=2).to(self.device)
