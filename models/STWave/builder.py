import numpy as np
import torch
from utils.base_builder import build_model
from utils.dataprocessing import load_adj, preprocess_adj, load_origin_dtw
import argparse
import os
import torch.nn as nn
from utils.loss import *
from .criterion import loss_function
from functools import partial

class build_stwave(build_model):
    def _get_model_parameter(self):
        configs                 = argparse.ArgumentParser()

        configs.num_nodes           = self.args['data_args']['num_of_vertices']
        configs.input_dims = self.args['data_args']['num_of_features']
        configs.output_dims = self.args['data_args']['num_of_features']
        configs.input_len = self.args['data_args']['num_of_history']
        configs.output_len = self.args['data_args']['num_of_predict']

        configs.layers              = self.args['base_model_args']['layers']
        configs.heads               = self.args['base_model_args']['heads']
        configs.dims                = self.args['base_model_args']['dims']
        configs.samples             = self.args['base_model_args']['samples']

        configs.adjtype             = 'eigen'
        configs.sigma1 = self.args['base_model_args']['sigma1']
        configs.sigma2 = self.args['base_model_args']['sigma2']
        configs.thres1 = self.args['base_model_args']['thres1']
        configs.thres2 = self.args['base_model_args']['thres2']

        adj_path = self.args['data_args']['adj_path']
        adj_mx, _ = load_adj(adj_path, configs.num_nodes)
        adj_mx = adj_mx + np.eye(adj_mx.shape[0])
        spawave = preprocess_adj((adj_mx, configs.heads*configs.dims), configs.adjtype)
        configs.spawave = spawave[0]

        tem_mx = np.asarray(load_origin_dtw(dtw_filename=self.args['base_model_args']['dtw_path']))
        tem_mx = tem_mx + np.eye(tem_mx.shape[0])
        temwave = preprocess_adj((tem_mx, configs.heads*configs.dims), configs.adjtype)
        configs.temwave = temwave[0]

        localadj = preprocess_adj(adj_mx, 'neighbors')
        configs.localadj = localadj[0]

        return configs

    def _build_model(self):
        configs = self._get_model_parameter()

        self.model = self.network(
            input_dims = configs.input_dims,
            output_dims=configs.output_dims,
            heads = configs.heads,
            dims = configs.dims,
            layers = configs.layers,
            samples = configs.samples,
            localadj = configs.localadj,
            spawave = configs.spawave,
            temwave = configs.temwave,
            input_len = configs.input_len,
            output_len = configs.output_len
        )

    def _build_criterion(self):
        assert self.criterion_type in ['L1', 'MSE', 'Masked_MAE']
        if self.criterion_type == 'L1':
            criterion = nn.L1Loss()
        elif self.criterion_type == 'MSE':
            criterion = nn.MSELoss()
        elif self.criterion_type == 'Masked_MAE':
            criterion = partial(masked_mae_torch, null_val=0)
        self.criterion = loss_function(self.args['base_model_args']['wave'],
                                       self.args['base_model_args']['level'],
                                       criterion)

    def build(self, data_loader, get_data_func):
        self._build_model()

        self.model = self.model.to(self.device)
        if self.use_multi_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)

        if (self.checkpoint_filename is not None) and (os.path.exists(self.checkpoint_filename)):
            print('loading checkpoint from {} ...'.format(self.checkpoint_filename))
            self._setup_graph(data_loader, get_data_func)
            self.model.load_state_dict(torch.load(self.checkpoint_filename))
            # shutil.copy(self.checkpoint_filename, self.cur_save_path)
        else:
            self._init_weight()

        if self.train_phase == 'adjust':
            for (name, param) in self.model.named_parameters():
                if 'end_emb' in name:
                    pass
                else:
                    param.requires_grad = False

        self._build_optimizer()
        self._build_schedular()
        self._build_criterion()
