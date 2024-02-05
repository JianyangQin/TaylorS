import torch
import torch.nn as nn
from .loss import *
import os
from functools import partial

class build_model(nn.Module):
    def __init__(self, network, train_phase, checkpoint_filename, **args):
        super(build_model, self).__init__()
        self.device = args['device']
        self.device_ids = args['device_ids']
        self.use_multi_gpu = args['use_multi_gpu']

        self.checkpoint_filename = checkpoint_filename
        self.train_phase = train_phase
        self.network = network
        self.args = args

        # training parameters
        self.criterion_type = args['base_train_args']['criterion_type']
        self.optimization = args['base_train_args']['optimization']

        if self.train_phase == 'train':
            self.lr = args['base_train_args']['train_lr']
        else:
            self.lr = args['base_train_args']['finetune_lr']
        self.weight_decay = args['base_train_args']['weight_decay']
        self.epsilon = args['base_train_args']['epsilon']

        self.lr_decay_type = args['base_train_args']['lr_decay']
        self.lr_decay_step = args['base_train_args']['lr_decay_step']
        self.lr_decay_rate = args['base_train_args']['lr_decay_ratio']

        self.clip = args['base_train_args']['clip']

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

    def _get_model_parameter(self):
        return None

    def _build_model(self):
        model_parameters = self._get_model_parameter()
        assert model_parameters is not None
        self.model = self.network(model_parameters)

    def _build_optimizer(self):
        assert self.model is not None
        if self.optimization == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=self.lr,
                                               weight_decay=self.weight_decay,
                                               eps=self.epsilon)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.lr,
                                              weight_decay=self.weight_decay,
                                              eps=self.epsilon)

    def _build_schedular(self):
        assert self.optimizer is not None
        if self.lr_decay_type == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=self.lr_decay_step,
                                                                  gamma=self.lr_decay_rate)
        elif self.lr_decay_type == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=self.lr_decay_step,
                                                             gamma=self.lr_decay_rate)
        elif self.lr_decay_type == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        mode='min',
                                                                        factor=self.lr_decay_rate,
                                                                        patience=self.lr_decay_step,
                                                                        threshold=1e-3,
                                                                        threshold_mode='rel',
                                                                        min_lr=2e-6,
                                                                        verbose=False)
        else:
            self.scheduler = None

    def _build_criterion(self):
        assert self.criterion_type in ['L1', 'MSE', 'SmoothL1', 'MAE', 'MSE', 'RMSE', 'MAPE',
                                       'Masked_MAE', 'Masked_MSE', 'Masked_RMSE', 'Masked_MAPE',
                                       'Logcosh', 'Huber', 'Quantile', 'R2', 'Evar']
        if self.criterion_type == 'L1':
            self.criterion = nn.L1Loss()
        elif self.criterion_type == 'MSE':
            self.criterion = nn.MSELoss()
        elif self.criterion_type == 'SmoothL1':
            self.criterion = nn.SmoothL1Loss()
        if self.criterion_type == 'MAE':
            self.criterion = masked_mae_torch
        elif self.criterion_type == 'MSE':
            self.criterion = masked_mse_torch
        elif self.criterion_type == 'RMSE':
            self.criterion = masked_rmse_torch
        elif self.criterion_type == 'MAPE':
            self.criterion = masked_mape_torch
        elif self.criterion_type == 'Logcosh':
            self.criterion = log_cosh_loss
        elif self.criterion_type == 'Huber':
            self.criterion = huber_loss
        elif self.criterion_type == 'Quantile':
            self.criterion = quantile_loss
        elif self.criterion_type == 'Masked_MAE':
            self.criterion = partial(masked_mae_torch, null_val=0)
        elif self.criterion_type == 'Masked_MSE':
            self.criterion = partial(masked_mse_torch, null_val=0)
        elif self.criterion_type == 'Masked_RMSE':
            self.criterion = partial(masked_rmse_torch, null_val=0)
        elif self.criterion_type == 'Masked_MAPE':
            self.criterion = partial(masked_mape_torch, null_val=0)
        elif self.criterion_type == 'R2':
            self.criterion = r2_score_torch
        elif self.criterion_type == 'Evar':
            self.criterion = explained_variance_score_torch
        else:
            self.criterion = nn.L1Loss()

    def _init_weight(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def _setup_graph(self, data_loader, get_data_func):
        pass

    def _print_parameter_number(self):
        param_size = 0
        param_sum = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
            param_sum += param.nelement()
        buffer_size = 0
        buffer_sum = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            buffer_sum += buffer.nelement()
        all_size = (param_size + buffer_size) / 1024 / 1024
        param_size = param_size / 1024 / 1024
        buffer_size = buffer_size / 1024 / 1024
        print('param size：{:.3f}MB'.format(param_size))
        print('buffer size：{:.3f}MB'.format(buffer_size))
        print('all size：{:.3f}MB'.format(all_size))
        return (param_size, buffer_size, all_size)

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
                if 'out_fc' in name:
                    pass
                else:
                    param.requires_grad = False

        self._print_parameter_number()

        self._build_optimizer()
        self._build_schedular()
        self._build_criterion()
