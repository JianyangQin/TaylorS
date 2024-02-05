import torch
import torch.nn as nn
from utils.base_builder import build_model
from utils.dataprocessing import load_adj
from .lib import transition_matrix
import os

class build_d2stgnn(build_model):
    def _get_model_parameter(self):
        configs                     = {}

        configs['device']           = self.device

        configs['num_nodes']        = self.args['data_args']['num_of_vertices']
        configs['seq_length']       = self.args['data_args']['num_of_predict']
        configs['num_feat']         = self.args['data_args']['num_of_features']
        configs['time_per_day']     = self.args['data_args']['time_per_day']

        configs['num_hidden']       = self.args['base_model_args']['num_hidden']
        configs['node_hidden']      = self.args['base_model_args']['node_hidden']
        configs['time_emb_dim']     = self.args['base_model_args']['time_emb_dim']
        configs['dropout']          = self.args['base_model_args']['dropout']

        configs['k_t']              = self.args['base_model_args']['k_t']
        configs['k_s']              = self.args['base_model_args']['k_s']
        configs['gap']              = self.args['base_model_args']['gap']
        configs['num_modalities']   = self.args['base_model_args']['num_modalities']

        adj_path = self.args['data_args']['adj_path']
        adj_ori, _ = load_adj(adj_path, configs['num_nodes'])
        adj_mx = [transition_matrix(adj_ori).T, transition_matrix(adj_ori.T).T]
        configs['adjs'] = [torch.tensor(i).to(self.device) for i in adj_mx]
        configs['adjs_ori'] = torch.tensor(adj_ori).to(self.device)

        return configs

    def _build_model(self):
        configs = self._get_model_parameter()

        self.model = self.network(**configs).to(self.device)

    def build(self, data_loader, get_data_func):
        self._build_model()

        self.model = self.model.to(self.device)
        if self.use_multi_gpu:
           self.model = nn.DataParallel(self.model, device_ids=self.device_ids)

        if (self.checkpoint_filename is not None) and (os.path.exists(self.checkpoint_filename)):
           print('loading checkpoint from {} ...'.format(self.checkpoint_filename))
           self._setup_graph(data_loader, get_data_func)
           self.model.load_state_dict(torch.load(self.checkpoint_filename, map_location=self.device))

        if self.train_phase == 'adjust':
            for (name, param) in self.model.named_parameters():
                if 'out_fc' in name:
                    pass
                else:
                    param.requires_grad = False

        self._build_optimizer()
        self._build_schedular()
        self._build_criterion()