import torch
from utils.base_builder import build_model
from utils.dataprocessing import load_spatial, load_dtw
import argparse
import os
import torch.nn as nn

class build_stgode(build_model):
    def _get_model_parameter(self):
        configs                 = argparse.ArgumentParser()

        configs.sp_filename     = self.args['data_args']['adj_path']
        configs.dtw_filename    = self.args['base_model_args']['dtw_path']

        configs.num_timesteps_input  = self.args['data_args']['num_of_predict']
        configs.num_timesteps_output  = self.args['data_args']['num_of_history']
        configs.num_nodes = self.args['data_args']['num_of_vertices']
        configs.num_features = self.args['data_args']['num_of_features']

        configs.sigma1        = self.args['base_model_args']['sigma1']
        configs.sigma2        = self.args['base_model_args']['sigma2']
        configs.thres1        = self.args['base_model_args']['thres1']
        configs.thres2        = self.args['base_model_args']['thres2']

        return configs

    def _build_model(self):
        configs = self._get_model_parameter()

        A_sp_wave = load_spatial(sp_filename=configs.sp_filename,
                                 num_of_vertices=configs.num_nodes,
                                 sigma=configs.sigma2,
                                 thres=configs.thres2)
        A_se_wave = load_dtw(dtw_filename=configs.dtw_filename,
                             sigma=configs.sigma1,
                             thres=configs.thres1)

        A_sp_wave = torch.from_numpy(A_sp_wave)
        A_se_wave = torch.from_numpy(A_se_wave)

        self.model = self.network(num_nodes=configs.num_nodes,
                                  num_features=configs.num_features,
                                  num_timesteps_input=configs.num_timesteps_input,
                                  num_timesteps_output=configs.num_timesteps_output,
                                  A_sp_hat=A_sp_wave,
                                  A_se_hat=A_se_wave)

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
                if 'pred' in name:
                    pass
                else:
                    param.requires_grad = False

        self._build_optimizer()
        self._build_schedular()
        self._build_criterion()