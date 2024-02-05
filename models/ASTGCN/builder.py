import torch
import torch.nn as nn
from utils.base_builder import build_model
from utils.dataprocessing import load_adj
import argparse
from .ASTGCN_r import scaled_Laplacian, cheb_polynomial
import os

class build_astgcn(build_model):
    def _get_model_parameter(self):
        configs                 = argparse.ArgumentParser()
        configs.adj_filename    = self.args['data_args']['adj_path']
        configs.num_of_predict  = self.args['data_args']['num_of_predict']
        configs.num_of_history  = self.args['data_args']['num_of_history']
        configs.num_of_vertices = self.args['data_args']['num_of_vertices']

        configs.nb_block        = self.args['base_model_args']['nb_block']
        configs.in_channels     = self.args['data_args']['num_of_features']
        configs.K               = self.args['base_model_args']['K']
        configs.nb_chev_filter  = self.args['base_model_args']['nb_chev_filter']
        configs.nb_time_filter  = self.args['base_model_args']['nb_time_filter']
        configs.time_strides    = self.args['base_model_args']['time_strides']
        return configs

    def _build_model(self):
        configs = self._get_model_parameter()

        adj_mx, distance_mx = load_adj(configs.adj_filename, configs.num_of_vertices, None)
        L_tilde = scaled_Laplacian(adj_mx)
        cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor)
                            for i in cheb_polynomial(L_tilde, configs.K)]
        self.model = self.network(
            nb_block=configs.nb_block,
            in_channels=configs.in_channels,
            K=configs.K,
            nb_chev_filter=configs.nb_chev_filter,
            nb_time_filter=configs.nb_time_filter,
            time_strides=configs.time_strides,
            cheb_polynomials=cheb_polynomials,
            num_for_predict=configs.num_of_predict,
            len_input=configs.num_of_history,
            num_of_vertices=configs.num_of_vertices
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
        else:
            self._init_weight()

        if self.train_phase == 'adjust':
            for (name, param) in self.model.named_parameters():
                if 'final_conv' in name:
                    pass
                else:
                    param.requires_grad = False

        self._build_optimizer()
        self._build_schedular()
        self._build_criterion()