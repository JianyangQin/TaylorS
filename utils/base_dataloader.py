import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from utils.tools import StandardScalar, MinMaxScalar, NormalScalar
from utils.timefeatures import get_timestamp, convert_array2timestamp

class load_base_data(nn.Module):
    def __init__(self, **args):
        super(load_base_data, self).__init__()
        self.args = args
        self.device = args['device']
        self.use_multi_gpu = args['use_multi_gpu']

        self.dataset = args['data_args']['dataset']
        self.dataset_path = args['data_args']['data_path']

        self.batch_size = args['data_args']['batch_size']
        self.train_ratio = args['data_args']['train_ratio']
        self.val_ratio = args['data_args']['val_ratio']
        self.test_ratio = args['data_args']['test_ratio']
        self.imputate_ratio = args['data_args']['imputate_ratio']

        self.num_of_history = args['data_args']['num_of_history']
        self.num_of_predict = args['data_args']['num_of_predict']
        self.num_of_vertices = args['data_args']['num_of_vertices']
        self.num_of_features = args['data_args']['num_of_features']

        self.start_time = args['data_args']['start_time']
        self.freq = args['data_args']['sample_freq']
        self.time_per_day = args['data_args']['time_per_day']
        self.time_embed = args['data_args']['time_embedding']
        self.time_features = args['data_args']['time_features']

        self.scalar = args['data_args']['scalar']
        self.data_scalar = None

    def _data_seq2instance(self, data, len_his, len_pred):
        num_step, num_of_vertices, dims = data.shape
        num_sample = num_step - len_his - len_pred + 1
        x = np.zeros(shape=(num_sample, len_his, num_of_vertices, dims))
        y = np.zeros(shape=(num_sample, len_pred, num_of_vertices, dims))
        for i in range(num_sample):
            x[i] = data[i: i + len_his]
            y[i] = data[i + len_his: i + len_his + len_pred]
        return x, y

    def _timestamp_seq2instance(self, data, num_of_history, num_of_predict):
        num_step, dims = data.shape
        num_sample = num_step - num_of_history - num_of_predict + 1
        x = np.zeros(shape=(num_sample, num_of_history, dims), dtype=np.float)
        y = np.zeros(shape=(num_sample, num_of_predict, dims), dtype=np.float)
        for i in range(num_sample):
            x[i] = data[i: i + num_of_history]
            y[i] = data[i + num_of_history: i + num_of_history + num_of_predict]
        return x, y

    def _get_raw_data(self):
        if self.dataset in ['PEMS04', 'PEMS08', 'Era5', 'Wind', 'Windmill']:
            raw_data = np.load(self.dataset_path)['data']
        elif self.dataset in ['METR-LA', 'PEMS-BAY']:
            raw_data = np.asarray(pd.read_hdf(self.dataset_path))
        elif self.dataset in ['HZ-METRO', 'SH-METRO']:
            raw_data = []
            for ds_path in self.dataset_path:
                with open(ds_path, 'rb') as f:
                    data = pickle.load(f)
                raw_data.append(data)
        if isinstance(raw_data, list) is False:
            if raw_data.ndim == 2:
                raw_data = np.expand_dims(raw_data, axis=-1)
            else:
                raw_data = raw_data[..., :self.num_of_features]
        return raw_data

    def _split_data(self, data):
        if self.dataset in ['HZ-METRO', 'SH-METRO']:
            train, val, test = data[0], data[1], data[2]
            train_x, train_y = np.transpose(train['x'], (0, 2, 3, 1)), np.transpose(train['y'], (0, 2, 3, 1))
            val_x, val_y = np.transpose(val['x'], (0, 2, 3, 1)), np.transpose(val['y'], (0, 2, 3, 1))
            test_x, test_y = np.transpose(test['x'], (0, 2, 3, 1)), np.transpose(test['y'], (0, 2, 3, 1))
        else:
            x, y = self._data_seq2instance(data, self.num_of_history, self.num_of_predict)
            x = np.transpose(x, (0, 2, 3, 1))
            y = np.transpose(y, (0, 2, 3, 1))

            len_total = len(x)
            len_train = round(len_total * self.train_ratio)
            len_test = round(len_total * self.test_ratio)
            len_val = len_total - len_train - len_test

            train_x, train_y = x[:len_train], y[:len_train]
            val_x, val_y = x[len_train:len_train+len_val], y[len_train:len_train+len_val]
            test_x, test_y = x[-len_test:], y[-len_test:]

        train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
        train_y = torch.from_numpy(train_y).type(torch.FloatTensor)

        val_x = torch.from_numpy(val_x).type(torch.FloatTensor)
        val_y = torch.from_numpy(val_y).type(torch.FloatTensor)

        test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
        test_y = torch.from_numpy(test_y).type(torch.FloatTensor)

        return (train_x, train_y, val_x, val_y, test_x, test_y)

    def _scale_data(self, data):
        train_x, train_y, val_x, val_y, test_x, test_y = data

        if self.scalar == 'standard':
            mean_val = train_x.mean(axis=(0, 1, 3), keepdims=True)
            std_val = train_x.std(axis=(0, 1, 3), keepdims=True)
            data_scalar = StandardScalar(mean=mean_val, std=std_val)
        elif self.scalar == 'minmax':
            max_val = train_x.max(axis=(0, 1, 3), keepdims=True)
            min_val = train_x.min(axis=(0, 1, 3), keepdims=True)
            data_scalar = MinMaxScalar(min=max_val, max=min_val)
        elif self.scalar == 'normal':
            max_val = train_x.max(axis=(0, 1, 3), keepdims=True)
            min_val = train_x.min(axis=(0, 1, 3), keepdims=True)
            data_scalar = NormalScalar(min=max_val, max=min_val)
        else:
            raise ValueError('scalar error!')

        train_x = data_scalar.transform(train_x)
        train_y = data_scalar.transform(train_y)

        val_x = data_scalar.transform(val_x)
        val_y = data_scalar.transform(val_y)

        test_x = data_scalar.transform(test_x)
        test_y = data_scalar.transform(test_y)

        self.data_scalar = data_scalar

        return (train_x, train_y, val_x, val_y, test_x, test_y), data_scalar

    def _get_timestamp(self, data):
        if self.dataset in ['HZ-METRO', 'SH-METRO']:
            train, val, test = data[0], data[1], data[2]
            train_tx, train_ty = train['xtime'], train['ytime']
            val_tx, val_ty = val['xtime'], val['ytime']
            test_tx, test_ty = test['xtime'], test['ytime']

            train_tx = convert_array2timestamp(train_tx, self.time_embed, self.time_features, self.freq)
            train_ty = convert_array2timestamp(train_ty, self.time_embed, self.time_features, self.freq)

            val_tx = convert_array2timestamp(val_tx, self.time_embed, self.time_features, self.freq)
            val_ty = convert_array2timestamp(val_ty, self.time_embed, self.time_features, self.freq)

            test_tx = convert_array2timestamp(test_tx, self.time_embed, self.time_features, self.freq)
            test_ty = convert_array2timestamp(test_ty, self.time_embed, self.time_features, self.freq)
            timestamp = (train_tx, train_ty, val_tx, val_ty, test_tx, test_ty)
            time_per_day = self.time_per_day
        else:
            len_data = len(data)
            timestamp, time_per_day = get_timestamp(embed=self.time_embed,
                                                    features=self.time_features,
                                                    start_time=self.start_time,
                                                    num_of_steps=len_data,
                                                    freq=self.freq,
                                                    num_of_history=self.num_of_history,
                                                    num_of_predict=self.num_of_predict,
                                                    train_ratio=self.train_ratio,
                                                    val_ratio=self.val_ratio,
                                                    test_ratio=self.test_ratio)

        return timestamp, time_per_day

    def _split_timestamp(self, data):
        if self.dataset in ['HZ-METRO', 'SH-METRO']:
            train_tx, train_ty, val_tx, val_ty, test_tx, test_ty = data
            train_ts = np.concatenate([train_tx, train_ty], axis=1)
            val_ts = np.concatenate([val_tx, val_ty], axis=1)
            test_ts = np.concatenate([test_tx, test_ty], axis=1)
        else:
            timestamp = self._timestamp_seq2instance(data, self.num_of_history, self.num_of_predict)
            timestamp = np.concatenate(timestamp, axis=1).astype(np.float)

            len_total = len(timestamp)
            len_train = round(len_total * self.train_ratio)
            len_test = round(len_total * self.test_ratio)
            len_val = len_total - len_train - len_test

            train_ts = timestamp[:len_train]
            val_ts = timestamp[len_train:len_train + len_val]
            test_ts = timestamp[-len_test:]

        train_ts = torch.from_numpy(train_ts).type(torch.FloatTensor)
        val_ts = torch.from_numpy(val_ts).type(torch.FloatTensor)
        test_ts = torch.from_numpy(test_ts).type(torch.FloatTensor)

        return (train_ts, val_ts, test_ts)

    def _imputate_data(self, data, ratio):
        data_size = data.size
        imputate_size = data_size * ratio
        idx = np.arange(0, data_size)
        imputate_idx = np.random.choice(idx, size=imputate_size, replace=False)
        data[imputate_idx] = 0.
        return data

    def load_data(self):
        raw_data = self._get_raw_data()
        data = self._split_data(raw_data)
        data, data_scalar = self._scale_data(data)
        timestamp, time_per_day = self._get_timestamp(raw_data)
        timestamp = self._split_timestamp(timestamp)

        train_x, train_y, val_x, val_y, test_x, test_y = data
        if self.imputate_ratio > 0:
            train_x = self._imputate_data(train_x, self.imputate_ratio)
        train_ts, val_ts, test_ts = timestamp

        train_dataset = torch.utils.data.TensorDataset(train_x, train_y, train_ts)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # ------- val_loader -------
        val_dataset = torch.utils.data.TensorDataset(val_x, val_y, val_ts)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # ------- test_loader -------
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y, test_ts)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print('train:', train_x.size(), train_y.size())
        print('val:', val_x.size(), val_y.size())
        print('test:', test_x.size(), test_y.size())

        return train_loader, val_loader, test_loader, data_scalar

    def get_train_data(self, batch):
        x, _, _ = batch
        return x.to(self.device)

    def get_finetune_data(self, batch):
        x, _, _ = batch
        return x.to(self.device)

    def get_adjust_data(self, batch):
        x, _, ts = batch
        return x.to(self.device), ts.to(self.device)

    def get_ground_truth(self, batch):
        _, y, _ = batch
        return y.to(self.device)



class load_auxiliary_data(nn.Module):
    def __init__(self, gpu_device, **args):
        super(load_auxiliary_data, self).__init__()
        self.device = gpu_device

        self.args = args

        self.dataset = args['data_args']['dataset']
        self.dataset_path = args['data_args']['data_path']

        self.batch_size = args['data_args']['batch_size']
        self.train_ratio = args['data_args']['train_ratio']
        self.val_ratio = args['data_args']['val_ratio']
        self.test_ratio = args['data_args']['test_ratio']
        self.imputate_ratio = args['data_args']['imputate_ratio']

        self.num_of_history = args['data_args']['num_of_history']
        self.num_of_predict = args['data_args']['num_of_predict']
        self.num_of_vertices = args['data_args']['num_of_vertices']
        self.num_of_features = args['data_args']['num_of_features']

        self.start_time = args['data_args']['start_time']
        self.freq = args['data_args']['sample_freq']
        self.time_per_day = args['data_args']['time_per_day']
        self.time_embed = args['data_args']['time_embedding']
        self.time_features = args['data_args']['time_features']

        self.scalar = args['data_args']['scalar']
        self.data_scalar = None

    def _data_seq2instance(self, data, len_his, len_pred):
        num_step, num_of_vertices, dims = data.shape
        num_sample = num_step - len_his - len_pred + 1
        x = np.zeros(shape=(num_sample, len_his, num_of_vertices, dims))
        y = np.zeros(shape=(num_sample, len_pred, num_of_vertices, dims))
        for i in range(num_sample):
            x[i] = data[i: i + len_his]
            y[i] = data[i + len_his: i + len_his + len_pred]
        return x, y

    def _timestamp_seq2instance(self, data, num_of_history, num_of_predict):
        num_step, dims = data.shape
        num_sample = num_step - num_of_history - num_of_predict + 1
        x = np.zeros(shape=(num_sample, num_of_history, dims), dtype=np.float)
        y = np.zeros(shape=(num_sample, num_of_predict, dims), dtype=np.float)
        for i in range(num_sample):
            x[i] = data[i: i + num_of_history]
            y[i] = data[i + num_of_history: i + num_of_history + num_of_predict]
        return x, y

    def _get_raw_data(self):
        if self.dataset in ['PEMS03', 'PEMS04', 'PEMS08', 'Era5', 'Wind', 'Windmill']:
            raw_data = np.load(self.dataset_path)['data']
        elif self.dataset in ['METR-LA', 'PEMS-BAY']:
            raw_data = np.asarray(pd.read_hdf(self.dataset_path))
        elif self.dataset in ['HZ-METRO', 'SH-METRO']:
            raw_data = []
            for ds_path in self.dataset_path:
                with open(ds_path, 'rb') as f:
                    data = pickle.load(f)
                raw_data.append(data)
        elif self.dataset in ['Air-Quality']:
            raw_data = []
            for ds_path in self.dataset_path:
                data = np.load(ds_path)
                raw_data.append(data)
        if isinstance(raw_data, list) is False:
            if raw_data.ndim == 2:
                raw_data = np.expand_dims(raw_data, axis=-1)
            else:
                raw_data = raw_data[..., -self.num_of_features:]
        return raw_data

    def _split_data(self, data):
        if self.dataset in ['HZ-METRO', 'SH-METRO', 'Air-Quality']:
            train, val, test = data[0], data[1], data[2]
            train_x, train_y = np.transpose(train['x'], (0, 2, 3, 1)), np.transpose(train['y'], (0, 2, 3, 1))
            val_x, val_y = np.transpose(val['x'], (0, 2, 3, 1)), np.transpose(val['y'], (0, 2, 3, 1))
            test_x, test_y = np.transpose(test['x'], (0, 2, 3, 1)), np.transpose(test['y'], (0, 2, 3, 1))
        else:
            x, y = self._data_seq2instance(data, self.num_of_history, self.num_of_predict)
            x = np.transpose(x, (0, 2, 3, 1))
            y = np.transpose(y, (0, 2, 3, 1))

            len_total = len(x)
            len_train = round(len_total * self.train_ratio)
            len_test = round(len_total * self.test_ratio)
            len_val = len_total - len_train - len_test

            train_x, train_y = x[:len_train], y[:len_train]
            val_x, val_y = x[len_train:len_train+len_val], y[len_train:len_train+len_val]
            test_x, test_y = x[-len_test:], y[-len_test:]

        train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
        train_y = torch.from_numpy(train_y).type(torch.FloatTensor)

        val_x = torch.from_numpy(val_x).type(torch.FloatTensor)
        val_y = torch.from_numpy(val_y).type(torch.FloatTensor)

        test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
        test_y = torch.from_numpy(test_y).type(torch.FloatTensor)

        return (train_x, train_y, val_x, val_y, test_x, test_y)

    def _scale_data(self, data):
        train_x, train_y, val_x, val_y, test_x, test_y = data

        if self.scalar == 'standard':
            mean_val = train_x.mean(axis=(0, 1, 3), keepdims=True)
            std_val = train_x.std(axis=(0, 1, 3), keepdims=True)
            data_scalar = StandardScalar(mean=mean_val, std=std_val)
        elif self.scalar == 'minmax':
            max_val = train_x.max(axis=(0, 1, 3), keepdims=True)
            min_val = train_x.min(axis=(0, 1, 3), keepdims=True)
            data_scalar = MinMaxScalar(min=max_val, max=min_val)
        elif self.scalar == 'normal':
            max_val = train_x.max(axis=(0, 1, 3), keepdims=True)
            min_val = train_x.min(axis=(0, 1, 3), keepdims=True)
            data_scalar = NormalScalar(min=max_val, max=min_val)
        else:
            raise ValueError('scalar error!')

        train_x = data_scalar.transform(train_x)
        train_y = data_scalar.transform(train_y)

        val_x = data_scalar.transform(val_x)
        val_y = data_scalar.transform(val_y)

        test_x = data_scalar.transform(test_x)
        test_y = data_scalar.transform(test_y)

        self.data_scalar = data_scalar

        return (train_x, train_y, val_x, val_y, test_x, test_y), data_scalar

    def _get_timestamp(self, data):
        if self.dataset in ['HZ-METRO', 'SH-METRO']:
            train, val, test = data[0], data[1], data[2]
            train_tx, train_ty = train['xtime'], train['ytime']
            val_tx, val_ty = val['xtime'], val['ytime']
            test_tx, test_ty = test['xtime'], test['ytime']

            train_tx = convert_array2timestamp(train_tx, self.time_embed, self.time_features, self.freq)
            train_ty = convert_array2timestamp(train_ty, self.time_embed, self.time_features, self.freq)

            val_tx = convert_array2timestamp(val_tx, self.time_embed, self.time_features, self.freq)
            val_ty = convert_array2timestamp(val_ty, self.time_embed, self.time_features, self.freq)

            test_tx = convert_array2timestamp(test_tx, self.time_embed, self.time_features, self.freq)
            test_ty = convert_array2timestamp(test_ty, self.time_embed, self.time_features, self.freq)
            timestamp = (train_tx, train_ty, val_tx, val_ty, test_tx, test_ty)
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

            # timestamp = self._timestamp_seq2instance(timestamp, self.num_of_history, self.num_of_predict)
            # timestamp = np.concatenate(timestamp, axis=1).astype(np.float)

            # len_total = len(timestamp)
            # len_train = round(len_total * self.train_ratio)
            # len_test = round(len_total * self.test_ratio)
            # len_val = len_total - len_train - len_test
            #
            # train_ts = timestamp[:len_train]
            # val_ts = timestamp[len_train:len_train + len_val]
            # test_ts = timestamp[-len_test:]

            # train_tx, val_tx, test_tx = timestamp[:len_train], timestamp[len_train:len_train+len_val], timestamp[-(len_test+24):-24]
            # train_ty, val_ty, test_ty = timestamp[24:24+len_train], timestamp[24+len_train:24+len_train+len_val], timestamp[-len_test:]
            # timestamp = (train_tx, train_ty, val_tx, val_ty, test_tx, test_ty)
        else:
            len_data = len(data)
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

        return timestamp, time_per_day

    def _split_timestamp(self, data):
        if self.dataset in ['HZ-METRO', 'SH-METRO']:
            train_tx, train_ty, val_tx, val_ty, test_tx, test_ty = data
            train_ts = np.concatenate([train_tx, train_ty], axis=1)
            val_ts = np.concatenate([val_tx, val_ty], axis=1)
            test_ts = np.concatenate([test_tx, test_ty], axis=1)
        else:
            timestamp = self._timestamp_seq2instance(data, self.num_of_history, self.num_of_predict)
            timestamp = np.concatenate(timestamp, axis=1).astype(np.float)

            len_total = len(timestamp)
            len_train = round(len_total * self.train_ratio)
            len_test = round(len_total * self.test_ratio)
            len_val = len_total - len_train - len_test

            train_ts = timestamp[:len_train]
            val_ts = timestamp[len_train:len_train + len_val]
            test_ts = timestamp[-len_test:]

        train_ts = torch.from_numpy(train_ts).type(torch.FloatTensor)
        val_ts = torch.from_numpy(val_ts).type(torch.FloatTensor)
        test_ts = torch.from_numpy(test_ts).type(torch.FloatTensor)

        return (train_ts, val_ts, test_ts)

    def _get_timeinday(self, data):
        if self.dataset in ['HZ-METRO', 'SH-METRO']:
            train, val, test = data[0], data[1], data[2]
            train_txs, val_txs, test_txs = train['xtime'], val['xtime'], test['xtime']
            train_tys, val_tys, test_tys = train['ytime'], val['ytime'], test['ytime']

            train_txs = convert_array2timestamp(train_txs, 'discrete', 2, self.freq)
            val_txs = convert_array2timestamp(val_txs, 'discrete', 2, self.freq)
            test_txs = convert_array2timestamp(test_txs, 'discrete', 2, self.freq)

            train_tys = convert_array2timestamp(train_tys, 'discrete', 2, self.freq)
            val_tys = convert_array2timestamp(val_tys, 'discrete', 2, self.freq)
            test_tys = convert_array2timestamp(test_tys, 'discrete', 2, self.freq)

            train_tid = np.concatenate((train_txs[..., 1:2], train_tys[..., 1:2]), axis=1) / self.time_per_day
            val_tid = np.concatenate((val_txs[..., 1:2], val_tys[..., 1:2]), axis=1) / self.time_per_day
            test_tid = np.concatenate((test_txs[..., 1:2], test_tys[..., 1:2]), axis=1) / self.time_per_day

            train_tid = np.repeat(np.expand_dims(train_tid, 1), self.num_of_vertices, axis=1).transpose((0, 1, 3, 2))
            val_tid = np.repeat(np.expand_dims(val_tid, 1), self.num_of_vertices, axis=1).transpose((0, 1, 3, 2))
            test_tid = np.repeat(np.expand_dims(test_tid, 1), self.num_of_vertices, axis=1).transpose((0, 1, 3, 2))
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

            tid = timestamp[..., 1:2] / self.time_per_day

            tid_x, tid_y = self._timestamp_seq2instance(tid, self.num_of_history, self.num_of_predict)
            tid = np.concatenate((tid_x, tid_y), axis=1)
            tid = np.repeat(np.expand_dims(tid, 1), self.num_of_vertices, axis=1).transpose((0, 1, 3, 2))

            len_total = len(tid)
            len_train = round(len_total * self.train_ratio)
            len_test = round(len_total * self.test_ratio)
            len_val = len_total - len_train - len_test

            train_tid = tid[:len_train]
            val_tid = tid[len_train:len_train + len_val]
            test_tid = tid[-len_test:]
        else:
            len_data = len(data)
            timestamp, time_per_day = get_timestamp(
                embed='discrete',
                features=2,
                start_time=self.start_time,
                num_of_steps=len_data,
                freq=self.freq,
                num_of_history=self.num_of_history,
                num_of_predict=self.num_of_predict,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio
            )

            tid = timestamp[..., 1:2] / self.time_per_day

            tid_x, tid_y = self._timestamp_seq2instance(tid, self.num_of_history, self.num_of_predict)
            tid = np.concatenate((tid_x, tid_y), axis=1)
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

    def random_zero(tensor, prob):
        random_tensor = torch.rand(tensor.shape)
        tensor[random_tensor < prob] = 0
        return tensor

    def _imputate_data(self, data, prob):
        imputate_tensor = torch.rand(data.shape)
        data[imputate_tensor < prob] = 0.
        return data

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
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # ------- test_loader -------
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y, test_ts, test_tid)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print('train:', train_x.size(), train_y.size())
        print('val:', val_x.size(), val_y.size())
        print('test:', test_x.size(), test_y.size())

        return train_loader, val_loader, test_loader, data_scalar

    def get_train_data(self, batch):
        x, _, _, _ = batch
        return x.to(self.device)

    def get_finetune_data(self, batch):
        x, _, _, _ = batch
        return x.to(self.device)

    def get_adjust_data(self, batch):
        x, _, ts, tid = batch
        return x.to(self.device), ts.to(self.device), tid.to(self.device)

    def get_ground_truth(self, batch):
        _, y, _, _ = batch
        return y.to(self.device)










