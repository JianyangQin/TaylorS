import torch
import numpy as np
from fastdtw import fastdtw
import os
import yaml
import argparse
import pandas as pd
import pickle

def seq2instance(data, len_his, len_pred):
    num_step, num_of_vertices, dims = data.shape
    num_sample = num_step - len_his - len_pred + 1
    x = np.zeros(shape=(num_sample, len_his, num_of_vertices, dims))
    y = np.zeros(shape=(num_sample, len_pred, num_of_vertices, dims))
    for i in range(num_sample):
        x[i] = data[i: i + len_his]
        y[i] = data[i + len_his: i + len_his + len_pred]
    return x, y


def get_raw_data(dataset, base_path, data_filename, num_of_features):
    if dataset in ['PEMS04', 'PEMS08', 'Era5', 'Wind', 'Windmill']:
        data_path = os.path.join(base_path, data_filename)
        raw_data = np.load(data_path)['data']
    elif dataset in ['METR-LA', 'PEMS-BAY']:
        data_path = os.path.join(base_path, data_filename)
        raw_data = np.asarray(pd.read_hdf(data_path))
    elif dataset in ['HZ-METRO', 'SH-METRO']:
        raw_data = []
        for df_name in data_filename:
            data_path = os.path.join(base_path, df_name)
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            raw_data.append(data['x'][:, 0, :, :])
        raw_data = np.concatenate(raw_data, 0)
    else:
        raise ValueError('Cannot read data!')

    if isinstance(raw_data, list) is False:
        if raw_data.ndim == 2:
            raw_data = np.expand_dims(raw_data, axis=-1)
        else:
            raw_data = raw_data[..., :num_of_features]
    return raw_data


def generate_dtw(data, num_of_vertices, time_per_day):
    data_mean = np.mean([data[:, :, 0][time_per_day * i: time_per_day * (i + 1)] for i in range(data.shape[0] // time_per_day)],
                        axis=0)
    data_mean = data_mean.squeeze().T
    dtw_distance = np.zeros((num_of_vertices, num_of_vertices))
    for i in range(num_of_vertices):
        for j in range(i, num_of_vertices):
            dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
    for i in range(num_of_vertices):
        for j in range(i):
            dtw_distance[i][j] = dtw_distance[j][i]

    return dtw_distance

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='../configs/HZ-METRO/HZMETRO_stfgnn.yaml', type=str,
                        help="configuration file path")
    args = parser.parse_args()

    print('Read configuration file: %s' % (args.config))
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    dataset = configs['data_args']['dataset']
    data_filename = configs['data_args']['data_path']
    adj_filename = configs['data_args']['adj_path']
    dtw_path = configs['base_model_args']['dtw_path']

    num_of_history = configs['data_args']['num_of_history']
    num_of_predict = configs['data_args']['num_of_predict']
    num_of_vertices = configs['data_args']['num_of_vertices']
    num_of_features = configs['data_args']['num_of_features']
    time_per_day = configs['data_args']['time_per_day']

    train_ratio = configs['data_args']['train_ratio']
    test_ratio = configs['data_args']['test_ratio']

    abs_path = os.path.abspath(__file__)
    abs_path = abs_path.split('/')
    abs_path.pop(-1)
    abs_path.pop(-1)
    base_path = "/"
    for path in abs_path:
        base_path = os.path.join(base_path, path)
    print("base path: %s\n" % base_path)

    save_file = os.path.join(base_path, dtw_path)
    print("save path: %s\n" % save_file)

    print("Start generate DTW ......")

    data = get_raw_data(dataset, base_path, data_filename, num_of_features)

    dtw_distance = generate_dtw(data, num_of_vertices, time_per_day)
    np.save(save_file, dtw_distance)

    print("DTW finish")