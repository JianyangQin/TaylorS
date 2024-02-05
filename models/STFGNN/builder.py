import torch
import torch.nn as nn
from utils.base_builder import build_model
from utils.dataprocessing import load_adj, load_dtw
import numpy as np
import argparse
import os

class build_stfgnn(build_model):
    def _construct_adj_fusion(self, A, A_dtw, steps):
        '''
        construct a bigger adjacency matrix using the given matrix

        Parameters
        ----------
        A: np.ndarray, adjacency matrix, shape is (N, N)

        steps: how many times of the does the new adj mx bigger than A

        Returns
        ----------
        new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)

        ----------
        This is 4N_1 mode:

        [T, 1, 1, T
         1, S, 1, 1
         1, 1, S, 1
         T, 1, 1, T]

        '''

        N = len(A)
        adj = np.zeros([N * steps] * 2) # "steps" = 4 !!!

        for i in range(steps):
            if (i == 1) or (i == 2):
                adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A
            else:
                adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw
        #'''
        for i in range(N):
            for k in range(steps - 1):
                adj[k * N + i, (k + 1) * N + i] = 1
                adj[(k + 1) * N + i, k * N + i] = 1
        #'''
        adj[3 * N: 4 * N, 0:  N] = A_dtw #adj[0 * N : 1 * N, 1 * N : 2 * N]
        adj[0 : N, 3 * N: 4 * N] = A_dtw #adj[0 * N : 1 * N, 1 * N : 2 * N]

        adj[2 * N: 3 * N, 0 : N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
        adj[0 : N, 2 * N: 3 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
        adj[1 * N: 2 * N, 3 * N: 4 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
        adj[3 * N: 4 * N, 1 * N: 2 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]


        for i in range(len(adj)):
            adj[i, i] = 1

        return adj

    def _get_model_parameter(self):
        configs                             = argparse.ArgumentParser()

        configs.history                     = self.args['data_args']['num_of_history']
        configs.horizon                     = self.args['data_args']['num_of_predict']
        configs.num_of_vertices             = self.args['data_args']['num_of_vertices']
        configs.in_dim                      = self.args['data_args']['num_of_features']
        configs.out_dim                     = self.args['data_args']['num_of_features']

        configs.hidden_dims                 = self.args['base_model_args']['hidden_dims']
        configs.first_layer_embedding_size  = self.args['base_model_args']['first_layer_embedding_size']
        configs.out_layer_dim               = self.args['base_model_args']['out_layer_dim']
        configs.activation                  = self.args['base_model_args']['activation']
        configs.use_mask                    = self.args['base_model_args']['use_mask']
        configs.temporal_emb                = self.args['base_model_args']['temporal_emb']
        configs.spatial_emb                 = self.args['base_model_args']['spatial_emb']
        configs.seq_emb                     = self.args['base_model_args']['seq_emb']
        configs.seq_emb_dim                 = self.args['base_model_args']['seq_emb_dim']
        configs.strides                     = self.args['base_model_args']['strides']
        configs.sigma                       = self.args['base_model_args']['sigma']
        configs.thres                       = self.args['base_model_args']['thres']

        return configs

    def _build_model(self):
        configs = self._get_model_parameter()

        A_sp, _ = load_adj(adj_filename=self.args['data_args']['adj_path'],
                           num_of_vertices=configs.num_of_vertices,
                           id_filename=None)

        A_se = load_dtw(dtw_filename=self.args['base_model_args']['dtw_path'],
                        sigma=configs.sigma,
                        thres=configs.thres)
        A_se[A_se > 0.] = 1.

        adj = self._construct_adj_fusion(A_sp, A_se, configs.strides)
        adj = torch.from_numpy(adj).type(torch.FloatTensor)

        self.model = self.network(configs, adj)

    def build(self, data_loader, get_data_func):
        self._build_model()

        self.model = self.model.to(self.device)
        if self.use_multi_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)

        if (self.checkpoint_filename is not None) and (os.path.exists(self.checkpoint_filename)):
            print('loading checkpoint from {} ...'.format(self.checkpoint_filename))
            self._setup_graph(data_loader, get_data_func)
            self.model.load_state_dict(torch.load(self.checkpoint_filename))

        if self.train_phase == 'adjust':
            for (name, param) in self.model.named_parameters():
                if 'predictLayer' in name:
                    pass
                else:
                    param.requires_grad = False

        self._build_optimizer()
        self._build_schedular()
        self._build_criterion()