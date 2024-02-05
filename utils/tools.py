import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
import re
from .timefeatures import time_features

class StandardScalar(nn.Module):
    """
    Standard the input
    """

    def __init__(self, mean, std):
        super(StandardScalar, self).__init__()
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def inverse_transform(self, data):
        if isinstance(data, tuple):
            temp = []
            for d in data:
                if isinstance(d, tuple) or isinstance(d, list):
                    pass
                else:
                    d = (d * self.std.to(d[0].device)) + self.mean.to(d[0].device)
                temp.append(d)
            data = temp
            return data
        if isinstance(data, list):
            temp = []
            for d in data:
                if isinstance(d, tuple) or isinstance(d, list):
                    pass
                else:
                    d = (d * self.std.to(d[0].device)) + self.mean.to(d[0].device)
                temp.append(d)
            data = temp
            return data
        return (data * self.std.to(data.device)) + self.mean.to(data.device)
        
class MinMaxScalar(nn.Module):
    """
    Standard the input
    """

    def __init__(self, min, max):
        super(MinMaxScalar, self).__init__()
        self.min = min
        self.max = max

    def transform(self, data):
        return 2. * (data - self.min.to(data.device)) / (self.max.to(data.device) - self.min.to(data.device)) - 1.

    def inverse_transform(self, data):
        return (data + 1.) * (self.max.to(data.device) - self.min.to(data.device)) / 2. + self.min.to(data.device)


class NormalScalar(nn.Module):
    """
    Standard the input
    """

    def __init__(self, min, max):
        super(NormalScalar, self).__init__()
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min.to(data.device)) / (self.max - self.min.to(data.device))

    def inverse_transform(self, data):
        return data * (self.max.to(data.device) - self.min.to(data.device)) + self.min.to(data.device)


def find_params(path, search_name=None):
    file_list = os.listdir(path)

    params = None
    # find specified params file

    file_list.sort(key=lambda fn: os.path.getmtime(path + "/" + fn))
    file_list_len = len(file_list)
    for i in range(0, file_list_len):
        filename = file_list[file_list_len - i - 1]
        if filename.endswith('params'):
            if search_name is None:
                params = os.path.join(path, filename)
                break
            elif (search_name in filename):
                params = os.path.join(path, filename)
                break

    if params is None:
        raise ValueError("params does not exist")

    return params

def find_epoch(path):
    epoch = -1

    if path.endswith('params'):
        filename = path.split('/')[-1]
        filename = filename.split('.')[0]
        epoch = int(filename.split('_')[-1])

    if epoch < 0:
        raise ValueError("epoch value error")

    return epoch

def select_save_metric(metric, loss, mae, rmse, mape):
    if metric == 'MAE':
        return mae
    elif metric == 'RMSE':
        return rmse
    elif metric == 'MAPE':
        return mape
    else:
        return loss