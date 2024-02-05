import numpy as np
import torch
from utils.base_builder import build_model
from utils.dataprocessing import load_spatial, preprocess_adj
import argparse
import os
import torch.nn as nn

class build_gwnet(build_model):
    def _get_model_parameter(self):
        configs                 = argparse.ArgumentParser()

        configs.num_nodes           = self.args['data_args']['num_of_vertices']
        configs.in_dim = self.args['data_args']['num_of_features'] + 1
        configs.out_dim = self.args['data_args']['num_of_predict']

        configs.dropout             = self.args['base_model_args']['dropout']
        configs.residual_channels   = self.args['base_model_args']['nhid']
        configs.dilation_channels   = self.args['base_model_args']['nhid']
        configs.skip_channels       = self.args['base_model_args']['nhid'] * 8
        configs.end_channels        = self.args['base_model_args']['nhid'] * 16

        configs.adjtype = self.args['base_model_args']['adjtype']
        configs.gcn_bool = self.args['base_model_args']['gcn_bool']
        configs.aptonly = self.args['base_model_args']['aptonly']
        configs.addaptadj = self.args['base_model_args']['addaptadj']
        configs.randomadj = self.args['base_model_args']['randomadj']

        adj_path = self.args['data_args']['adj_path']
        adj_mx = np.asarray(load_spatial(adj_path, configs.num_nodes, 10, 0.5))
        adj_mx = preprocess_adj(adj_mx, configs.adjtype)
        supports = [torch.tensor(i) for i in adj_mx]
        if configs.randomadj:
            configs.adjinit = None
        else:
            configs.adjinit = supports[0]
        if configs.aptonly:
            supports = None
        configs.supports = supports

        return configs

    def _build_model(self):
        configs = self._get_model_parameter()

        self.model = self.network(
            num_nodes=configs.num_nodes,
            dropout=configs.dropout,
            supports=configs.supports,
            gcn_bool=configs.gcn_bool,
            addaptadj=configs.addaptadj,
            adjinit=configs.adjinit,
            in_dim=configs.in_dim,
            out_dim=configs.out_dim,
            residual_channels=configs.residual_channels,
            dilation_channels=configs.dilation_channels,
            skip_channels=configs.skip_channels,
            end_channels=configs.end_channels
        )

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
                if 'end_conv' in name:
                    pass
                else:
                    param.requires_grad = False

        self._build_optimizer()
        self._build_schedular()
        self._build_criterion()
