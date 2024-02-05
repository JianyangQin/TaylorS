import torch
import torch.nn as nn
from utils.base_builder import build_model
from .vector_fields import *
import argparse
import os

class build_stgncde(build_model):
    def _get_model_parameter(self):
        configs                 = argparse.ArgumentParser()

        configs.num_of_predict  = self.args['data_args']['num_of_predict']
        configs.num_of_history  = self.args['data_args']['num_of_history']
        configs.num_nodes       = self.args['data_args']['num_of_vertices']

        configs.input_dim       = self.args['data_args']['num_of_features'] + 1
        configs.output_dim      = self.args['data_args']['num_of_features']
        configs.embed_dim       = self.args['base_model_args']['embed_dim']
        configs.hid_dim         = self.args['base_model_args']['hid_dim']
        configs.hid_hid_dim     = self.args['base_model_args']['hid_hid_dim']
        configs.num_layers      = self.args['base_model_args']['num_layers']
        configs.default_graph   = self.args['base_model_args']['default_graph']
        configs.atol            = self.args['base_model_args']['atol']
        configs.rtol            = self.args['base_model_args']['rtol']
        configs.solver          = self.args['base_model_args']['solver']
        configs.cheb_k          = self.args['base_model_args']['cheb_k']
        configs.g_type          = self.args['base_model_args']['g_type']

        return configs

    def _build_model(self):
        configs = self._get_model_parameter()

        vector_field_f = FinalTanh_f(
            input_channels=configs.input_dim,
            hidden_channels=configs.hid_dim,
            hidden_hidden_channels=configs.hid_hid_dim,
            num_hidden_layers=configs.num_layers
        )

        vector_field_g = VectorField_g(
            input_channels=configs.input_dim,
            hidden_channels=configs.hid_dim,
            hidden_hidden_channels=configs.hid_hid_dim,
            num_hidden_layers=configs.num_layers,
            num_nodes=configs.num_nodes,
            cheb_k=configs.cheb_k,
            embed_dim=configs.embed_dim,
            g_type=configs.g_type
        )

        self.model = self.network(
            func_f=vector_field_f,
            func_g=vector_field_g,
            num_node=configs.num_nodes,
            horizon=configs.num_of_predict,
            num_layers=configs.num_layers,
            input_channels=configs.input_dim,
            hidden_channels=configs.hid_dim,
            output_channels=configs.output_dim,
            embed_dim=configs.embed_dim,
            default_graph=configs.default_graph,
            atol=configs.atol,
            rtol=configs.rtol,
            solver=configs.solver
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