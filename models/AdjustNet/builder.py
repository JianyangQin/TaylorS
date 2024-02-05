import torch
import torch.nn as nn
import numpy as np
from utils.base_builder import build_model
from utils.dataprocessing import load_adj, load_spatial, preprocess_adj
from .ASTGCN import scaled_Laplacian, cheb_polynomial
from .Criterion import loss_function
import argparse
import os

class build_adjustnet_astgcn(build_model):
    def __init__(self, network, train_phase, checkpoint_filename, **args):
        super(build_adjustnet_astgcn, self).__init__(network, train_phase, checkpoint_filename, **args)
        self.device = args['device']
        self.device_ids = args['device_ids']
        self.use_multi_gpu = args['use_multi_gpu']

        self.checkpoint_filename = checkpoint_filename
        self.train_phase = train_phase
        self.network = network
        self.args = args

        # training parameters
        self.criterion_type = args['adjust_train_args']['criterion_type']
        self.optimization = args['adjust_train_args']['optimization']

        self.lr = args['adjust_train_args']['lr']
        self.weight_decay = args['adjust_train_args']['weight_decay']
        self.epsilon = args['adjust_train_args']['epsilon']

        self.lr_decay_type = args['adjust_train_args']['lr_decay']
        self.lr_decay_step = args['adjust_train_args']['lr_decay_step']
        self.lr_decay_rate = args['adjust_train_args']['lr_decay_ratio']

        self.clip = args['adjust_train_args']['clip']

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

    def _get_model_parameter(self):
        configs = argparse.ArgumentParser()

        configs.device = self.device

        self.adj_filename = self.args['data_args']['adj_path']
        configs.num_of_vertices = self.args['data_args']['num_of_vertices']
        adj_mx, distance_mx = load_adj(self.adj_filename, configs.num_of_vertices, None)
        L_tilde = scaled_Laplacian(adj_mx)
        configs.cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor)
                                   for i in cheb_polynomial(L_tilde, self.args['adjust_train_args']['K'])]

        configs.batch_size = self.args['data_args']['batch_size']
        configs.num_of_history = self.args['data_args']['num_of_history']
        configs.num_of_predict = self.args['data_args']['num_of_predict']
        configs.num_of_layer = self.args['adjust_train_args']['num_of_layer']
        configs.time_per_day = self.args['data_args']['time_per_day']
        configs.freq = self.args['data_args']['sample_freq']
        configs.time_embedding = self.args['data_args']['time_embedding']
        configs.time_features = self.args['data_args']['time_features']

        # configs.freq = self.args['Adjust_model_args']['freq']
        configs.input_dim = self.args['data_args']['num_of_features']
        configs.hidden_dim = self.args['adjust_train_args']['hidden_dim']
        configs.output_dim = self.args['adjust_train_args']['output_dim']
        configs.mu_dim = self.args['adjust_train_args']['mu_dim']
        configs.num_of_head = self.args['adjust_train_args']['num_of_head']
        configs.bn_decay = self.args['adjust_train_args']['bn_decay']
        configs.K = self.args['adjust_train_args']['K']
        return configs

    def _build_criterion(self):
        assert self.criterion_type in ['L1', 'MSE', 'SmoothL1']
        if self.criterion_type == 'L1':
            criterion = nn.L1Loss()
        elif self.criterion_type == 'MSE':
            criterion = nn.MSELoss()
        elif self.criterion_type == 'SmoothL1':
            criterion = nn.SmoothL1Loss()
        self.criterion = loss_function(criterion)

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

        self._build_optimizer()
        self._build_schedular()
        self._build_criterion()


class build_adjustnet_gwnet(build_model):
    def __init__(self, network, train_phase, checkpoint_filename, **args):
        super(build_adjustnet_gwnet, self).__init__(network, train_phase, checkpoint_filename, **args)
        self.device = args['device']
        self.device_ids = args['device_ids']
        self.use_multi_gpu = args['use_multi_gpu']

        self.checkpoint_filename = checkpoint_filename
        self.train_phase = train_phase
        self.network = network
        self.args = args

        # training parameters
        self.criterion_type = args['adjust_train_args']['criterion_type']
        self.optimization = args['adjust_train_args']['optimization']

        self.lr = args['adjust_train_args']['lr']
        self.weight_decay = args['adjust_train_args']['weight_decay']
        self.epsilon = args['adjust_train_args']['epsilon']

        self.lr_decay_type = args['adjust_train_args']['lr_decay']
        self.lr_decay_step = args['adjust_train_args']['lr_decay_step']
        self.lr_decay_rate = args['adjust_train_args']['lr_decay_ratio']

        self.clip = args['adjust_train_args']['clip']

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

    def _get_model_parameter(self):
        configs = argparse.ArgumentParser()

        configs.device = self.device

        self.adj_filename = self.args['data_args']['adj_path']
        configs.num_of_vertices = self.args['data_args']['num_of_vertices']

        configs.batch_size = self.args['data_args']['batch_size']
        configs.num_of_history = self.args['data_args']['num_of_history']
        configs.num_of_predict = self.args['data_args']['num_of_predict']
        configs.time_per_day = self.args['data_args']['time_per_day']
        configs.freq = self.args['data_args']['sample_freq']
        configs.time_embedding = self.args['data_args']['time_embedding']
        configs.time_features = self.args['data_args']['time_features']

        # configs.freq = self.args['Adjust_model_args']['freq']
        configs.num_of_layer = self.args['adjust_model_args']['num_of_layer']
        configs.input_dim = self.args['data_args']['num_of_features']
        configs.hidden_dim = self.args['adjust_model_args']['hidden_dim']
        configs.output_dim = self.args['adjust_model_args']['output_dim']
        configs.mu_dim = self.args['adjust_model_args']['mu_dim']
        configs.num_of_head = self.args['adjust_model_args']['num_of_head']
        configs.bn_decay = self.args['adjust_model_args']['bn_decay']

        configs.num_nodes = self.args['data_args']['num_of_vertices']
        configs.in_dim = self.args['data_args']['num_of_features'] + 1
        configs.out_dim = self.args['data_args']['num_of_predict']

        configs.dropout = self.args['adjust_model_args']['dropout']
        configs.residual_channels = self.args['adjust_model_args']['nhid']
        configs.dilation_channels = self.args['adjust_model_args']['nhid']
        configs.skip_channels = self.args['adjust_model_args']['nhid'] * 8
        configs.end_channels = self.args['adjust_model_args']['nhid'] * 16

        configs.adjtype = self.args['adjust_model_args']['adjtype']
        configs.gcn_bool = self.args['adjust_model_args']['gcn_bool']
        configs.aptonly = self.args['adjust_model_args']['aptonly']
        configs.addaptadj = self.args['adjust_model_args']['addaptadj']
        configs.randomadj = self.args['adjust_model_args']['randomadj']

        adj_path = self.args['data_args']['adj_path']
        adj_mx = np.asarray(load_spatial(adj_path, configs.num_nodes, 10, 0.5))
        adj_mx = preprocess_adj(adj_mx, configs.adjtype)
        supports = [torch.tensor(i).to(self.device) for i in adj_mx]
        if configs.randomadj:
            configs.adjinit = None
        else:
            configs.adjinit = supports[0]
        if configs.aptonly:
            supports = None
        configs.supports = supports

        return configs

    def _build_criterion(self):
        assert self.criterion_type in ['L1', 'MSE', 'SmoothL1']
        if self.criterion_type == 'L1':
            criterion = nn.L1Loss()
        elif self.criterion_type == 'MSE':
            criterion = nn.MSELoss()
        elif self.criterion_type == 'SmoothL1':
            criterion = nn.SmoothL1Loss()
        self.criterion = loss_function(criterion)

    def build(self, data_loader, get_data_func):
        self._build_model()

        self.model = self.model.to(self.device)
        if self.use_multi_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)

        if (self.checkpoint_filename is not None) and (os.path.exists(self.checkpoint_filename)):
            print('loading checkpoint from {} ...'.format(self.checkpoint_filename))
            self._setup_graph(data_loader, get_data_func)
            self.model.load_state_dict(torch.load(self.checkpoint_filename))

        self._print_parameter_number()

        self._build_optimizer()
        self._build_schedular()
        self._build_criterion()
