import torch
import torch.nn as nn
from utils.base_builder import build_model
from utils.dataprocessing import load_se
import argparse
import os

class build_gman(build_model):
    def _get_model_parameter(self):
        configs = argparse.ArgumentParser()

        configs.device = self.device

        configs.input_dim = self.args['data_args']['num_of_features']
        configs.num_of_history = self.args['data_args']['num_of_history']

        configs.SE = torch.from_numpy(load_se(self.args['base_model_args']['se_path'])).type(torch.FloatTensor)
        configs.L = self.args['base_model_args']['L']
        configs.K = self.args['base_model_args']['K']
        configs.d = self.args['base_model_args']['d']
        configs.bn_decay = self.args['base_model_args']['bn_decay']

        return configs

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
                if 'end_conv' in name:
                    pass
                else:
                    param.requires_grad = False

        self._build_optimizer()
        self._build_schedular()
        self._build_criterion()