import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
import math
from .Embed import *
from .ASTGCN import ASTGCN
from .GraphWaveNet import GWNET

class TransformAttention(nn.Module):
    '''
    transform attention mechanism
    X:        [batch_size, num_his, num_vertex, D]
    STE_his:  [batch_size, num_his, num_vertex, D]
    STE_pred: [batch_size, num_pred, num_vertex, D]
    K:        number of attention heads
    d:        dimension of each attention outputs
    return:   [batch_size, num_pred, num_vertex, D]
    '''

    def __init__(self, hidden_dim, num_of_head, bn_decay):
        super(TransformAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_of_head = num_of_head

        assert hidden_dim % num_of_head == 0

        self.fc_q = DataEmbedding(input_dims=hidden_dim, units=hidden_dim, activations=F.relu, bn_decay=bn_decay)
        self.fc_k = DataEmbedding(input_dims=hidden_dim, units=hidden_dim, activations=F.relu, bn_decay=bn_decay)
        self.fc_v = DataEmbedding(input_dims=hidden_dim, units=hidden_dim, activations=None, bn_decay=bn_decay)
        self.fc = DataEmbedding(input_dims=hidden_dim, units=hidden_dim, activations=None, bn_decay=bn_decay)

    def forward(self, x, his, pred):
        batch_size = x.shape[0]

        # [batch_size, num_of_vertices, num_of_steps, k * d]
        query = self.fc_q(pred)
        key = self.fc_k(his)
        value = self.fc_v(x)

        # query: [k * batch_size, num_of_vertices, num_of_predict, d]
        query = torch.cat(torch.split(query, self.num_of_head, dim=2), dim=0).permute(0, 1, 3, 2)

        # key:   [k * batch_size, num_of_vertices, d, num_of_history]
        key = torch.cat(torch.split(key, self.num_of_head, dim=2), dim=0)

        # value: [k * batch_size, num_of_vertices, num_of_history, d]
        value = torch.cat(torch.split(value, self.num_of_head, dim=2), dim=0).permute(0, 1, 3, 2)

        # [k * batch_size, num_of_vertices, num_of_prediction, num_of_history]
        attention = torch.matmul(query, key)
        attention /= (self.hidden_dim ** 0.5)
        attention = F.softmax(attention, dim=-1)

        # [batch_size, num_of_vertices, d, num_of_prediction]
        y = torch.matmul(attention, value)
        y = torch.cat(torch.split(y, batch_size, dim=0), dim=-1).permute(0, 1, 3, 2)
        y = self.fc(y)

        return y


class AdjustNet_withASTGCN(nn.Module):
    def __init__(self, configs):
        super(AdjustNet_withASTGCN, self).__init__()
        self.device = configs.device
        self.batch_size = configs.batch_size
        self.num_of_vertices = configs.num_of_vertices
        self.num_of_history = configs.num_of_history
        self.num_of_predict = configs.num_of_predict
        self.num_of_layer = configs.num_of_layer
        self.input_dim = configs.input_dim
        self.hidden_dim = configs.hidden_dim
        self.output_dim = configs.output_dim
        self.num_of_head = configs.num_of_head
        self.mu_dim = configs.mu_dim
        self.bn_decay = configs.bn_decay
        self.time_per_day = configs.time_per_day
        self.freq = configs.freq
        self.time_embedding = configs.time_embedding
        self.time_features = configs.time_features
        self.threshold = nn.Parameter(torch.Tensor([1e-5]), requires_grad=False)

        assert self.hidden_dim % self.num_of_head == 0

        self.data_embedding = DataEmbedding(
            input_dims=[self.input_dim, self.hidden_dim],
            units=[self.hidden_dim, self.hidden_dim],
            activations=[F.relu, None],
            bn_decay=self.bn_decay
        )

        if configs.time_embedding == 'discrete':
            self.temporal_embedding = DiscreteTemporalEmbedding(
                features=self.time_features,
                channel=self.hidden_dim,
                num_of_vertices=self.num_of_vertices,
                time_per_day=self.time_per_day
            )
        else:
            self.temporal_embedding = ContinousTemporalEmbedding(
                channel=self.hidden_dim,
                freq=self.freq,
                num_of_vertices=self.num_of_vertices
            )

        # encoder and decoder for differential prediction
        self.encoder_dev = nn.ModuleList([ASTGCN(
            in_channels=self.hidden_dim,
            K=3,
            nb_chev_filter=self.hidden_dim,
            nb_time_filter=self.hidden_dim,
            time_strides=1,
            cheb_polynomials=configs.cheb_polynomials,
            num_of_vertices=self.num_of_vertices,
            num_of_timesteps=self.num_of_history-layer_idx
        ) for layer_idx in range(1, self.num_of_layer+1)])

        self.transform_dev = nn.ModuleList([TransformAttention(
            hidden_dim=self.hidden_dim,
            num_of_head=self.num_of_head,
            bn_decay=self.bn_decay
        ) for _ in range(1, self.num_of_layer+1)])

        self.prefixer_dev = nn.ModuleList([ASTGCN(
            in_channels=self.hidden_dim,
            K=3,
            nb_chev_filter=self.output_dim,
            nb_time_filter=self.output_dim,
            time_strides=1,
            cheb_polynomials=configs.cheb_polynomials,
            num_of_vertices=self.num_of_vertices,
            num_of_timesteps=self.num_of_predict-layer_idx
        ) for layer_idx in range(1, self.num_of_layer+1)])

        self.fc_dev = nn.ModuleList([nn.Conv2d(
            in_channels=self.num_of_predict - layer_idx,
            out_channels=self.input_dim * (self.num_of_predict - layer_idx),
            kernel_size=(1, self.hidden_dim)
        ) for layer_idx in range(1, self.num_of_layer+1)])

        # decoder for wave prediction
        self.transform_bias = nn.ModuleList([TransformAttention(
            hidden_dim=self.hidden_dim,
            num_of_head=self.num_of_head,
            bn_decay=self.bn_decay
        ) for _ in range(1, self.num_of_layer+1)])

        self.decoder_bias = nn.ModuleList([ASTGCN(
            in_channels=self.hidden_dim,
            K=3,
            nb_chev_filter=self.output_dim,
            nb_time_filter=self.output_dim,
            time_strides=1,
            cheb_polynomials=configs.cheb_polynomials,
            num_of_vertices=self.num_of_vertices,
            num_of_timesteps=self.num_of_predict
        ) for _ in range(1, self.num_of_layer+1)])

        self.fc_bias = nn.ModuleList([nn.Conv2d(
            in_channels=self.num_of_predict,
            out_channels=self.input_dim * self.num_of_predict,
            kernel_size=(1, self.output_dim)
        ) for _ in range(1, self.num_of_layer+1)])

        self.mu_logvar_dev = nn.ModuleList([nn.Sequential(
            nn.Linear((self.num_of_predict - layer_idx), self.mu_dim * 2),
            nn.Linear(self.mu_dim * 2, self.mu_dim * 2)
        ) for layer_idx in range(1, self.num_of_layer+1)])

        self.mu_logvar_bias = nn.ModuleList([nn.Sequential(
            nn.Linear(self.num_of_predict, self.mu_dim * 2),
            nn.Linear(self.mu_dim * 2, self.mu_dim * 2)
        ) for _ in range(1, self.num_of_layer+1)])

    def _get_mu_logvar_dev(self, x, idx):
        out = self.mu_logvar_dev[idx](x)
        mu, log_var = torch.split(out, [self.mu_dim, self.mu_dim], dim=-1)
        return mu, log_var

    def _get_mu_logvar_bias(self, x, idx):
        out = self.mu_logvar_bias[idx](x)
        mu, log_var = torch.split(out, [self.mu_dim, self.mu_dim], dim=-1)
        return mu, log_var

    def _get_diff(self, x, n, dim):
        assert n > 0
        for i in range(n):
            x = torch.diff(x, dim=dim)
        return x

    def forward(self, batch):
        taylor_bias, taylor_dev, taylor_dev_mu, taylor_dev_logvar, taylor_bias_mu, taylor_bias_logvar \
            = [], [], [], [], [], []

        for layer_idx in range(self.num_of_layer):
            x, t = batch

            x = self.data_embedding(x)

            t = self.temporal_embedding(t)
            t_his = t[:, :, :, :self.num_of_history]
            t_pred = t[:, :, :, self.num_of_history:]

            dx_his = self._get_diff(x, n=layer_idx+1, dim=-1)
            dt_his = self._get_diff(t_his, n=layer_idx+1, dim=-1)
            dt_his = torch.where(dt_his < self.threshold, self.threshold, dt_his)
            dt_pred = self._get_diff(t_pred, n=layer_idx+1, dim=-1)

            x = dx_his / dt_his

            dev_z = self.encoder_dev[layer_idx](x + dt_his)
            dev_z = self.transform_dev[layer_idx](dev_z, dt_his, dt_pred)
            dev_z = self.prefixer_dev[layer_idx](dev_z + dt_pred)
            dev = self.fc_dev[layer_idx](dev_z.permute(0, 3, 1, 2))
            dev = dev[:, :, :, -1].permute(0, 2, 1).reshape(-1, self.num_of_vertices, self.input_dim, self.num_of_predict-layer_idx-1)
            dev_mu, dev_logvar = self._get_mu_logvar_dev(dev, layer_idx)

            bias_z = self.transform_bias[layer_idx](dev_z, dt_pred, t_pred)
            bias_z = self.decoder_bias[layer_idx](bias_z + t_pred)
            bias = self.fc_bias[layer_idx](bias_z.permute(0, 3, 1, 2))
            bias = bias[:, :, :, -1].permute(0, 2, 1).reshape(-1, self.num_of_vertices, self.input_dim, self.num_of_predict)
            bias_mu, bias_logvar = self._get_mu_logvar_bias(bias, layer_idx)

            taylor_bias.append(bias)
            taylor_dev.append(dev)
            taylor_dev_mu.append(dev_mu)
            taylor_dev_logvar.append(dev_logvar)
            taylor_bias_mu.append(bias_mu)
            taylor_bias_logvar.append(bias_logvar)

        taylor_bias = torch.sum(torch.stack(taylor_bias), dim=0)

        return taylor_bias, taylor_dev, taylor_dev_mu, taylor_dev_logvar, taylor_bias_mu, taylor_bias_logvar

class AdjustNet_withGWNET(nn.Module):
    def __init__(self, configs):
        super(AdjustNet_withGWNET, self).__init__()
        self.device = configs.device
        self.batch_size = configs.batch_size
        self.num_of_vertices = configs.num_of_vertices
        self.num_of_history = configs.num_of_history
        self.num_of_predict = configs.num_of_predict
        self.num_of_layer = configs.num_of_layer
        self.input_dim = configs.input_dim
        self.hidden_dim = configs.hidden_dim
        self.output_dim = configs.output_dim
        self.num_of_head = configs.num_of_head
        self.mu_dim = configs.mu_dim
        self.bn_decay = configs.bn_decay
        self.time_per_day = configs.time_per_day
        self.freq = configs.freq
        self.time_embedding = configs.time_embedding
        self.time_features = configs.time_features
        self.threshold = nn.Parameter(torch.Tensor([1e-5]), requires_grad=False)

        assert self.hidden_dim % self.num_of_head == 0

        self.data_embedding = DataEmbedding(
            input_dims=[self.input_dim, self.hidden_dim],
            units=[self.hidden_dim, self.hidden_dim],
            activations=[F.relu, None],
            bn_decay=self.bn_decay
        )

        if configs.time_embedding == 'discrete':
            self.temporal_embedding = DiscreteTemporalEmbedding(
                features=self.time_features,
                channel=self.hidden_dim,
                num_of_vertices=self.num_of_vertices,
                time_per_day=self.time_per_day
            )
        else:
            self.temporal_embedding = ContinousTemporalEmbedding(
                channel=self.hidden_dim,
                freq=self.freq,
                num_of_vertices=self.num_of_vertices
            )

        # encoder and decoder for differential prediction
        self.encoder_dev = nn.ModuleList([GWNET(
            num_nodes=configs.num_nodes,
            dropout=configs.dropout,
            supports=configs.supports,
            gcn_bool=configs.gcn_bool,
            addaptadj=configs.addaptadj,
            adjinit=configs.adjinit,
            in_dim=configs.hidden_dim + 1,
            out_dim=configs.num_of_predict - layer_idx,
            residual_channels=configs.residual_channels,
            dilation_channels=configs.dilation_channels,
            skip_channels=configs.skip_channels,
            end_channels=configs.end_channels
        ) for layer_idx in range(1, self.num_of_layer+1)])

        self.transform_dev = nn.ModuleList([TransformAttention(
            hidden_dim=self.hidden_dim,
            num_of_head=self.num_of_head,
            bn_decay=self.bn_decay
        ) for _ in range(1, self.num_of_layer+1)])

        self.prefixer_dev = nn.ModuleList([GWNET(
            num_nodes=configs.num_nodes,
            dropout=configs.dropout,
            supports=configs.supports,
            gcn_bool=configs.gcn_bool,
            addaptadj=configs.addaptadj,
            adjinit=configs.adjinit,
            in_dim=configs.hidden_dim + 1,
            out_dim=configs.num_of_predict - layer_idx,
            residual_channels=configs.residual_channels,
            dilation_channels=configs.dilation_channels,
            skip_channels=configs.skip_channels,
            end_channels=configs.end_channels
        ) for layer_idx in range(1, self.num_of_layer+1)])

        self.fc_dev = nn.ModuleList([nn.Conv2d(
            in_channels=self.num_of_predict - layer_idx,
            out_channels=self.input_dim * (self.num_of_predict - layer_idx),
            kernel_size=(1, self.hidden_dim)
        ) for layer_idx in range(1, self.num_of_layer+1)])

        # decoder for wave prediction
        self.transform_bias = nn.ModuleList([TransformAttention(
            hidden_dim=self.hidden_dim,
            num_of_head=self.num_of_head,
            bn_decay=self.bn_decay
        ) for _ in range(1, self.num_of_layer+1)])

        self.decoder_bias = nn.ModuleList([GWNET(
            num_nodes=configs.num_nodes,
            dropout=configs.dropout,
            supports=configs.supports,
            gcn_bool=configs.gcn_bool,
            addaptadj=configs.addaptadj,
            adjinit=configs.adjinit,
            in_dim=configs.hidden_dim + 1,
            out_dim=configs.num_of_predict,
            residual_channels=configs.residual_channels,
            dilation_channels=configs.dilation_channels,
            skip_channels=configs.skip_channels,
            end_channels=configs.end_channels
        ) for _ in range(1, self.num_of_layer+1)])

        self.fc_bias = nn.ModuleList([nn.Conv2d(
            in_channels=self.num_of_predict,
            out_channels=self.input_dim * self.num_of_predict,
            kernel_size=(1, self.output_dim)
        ) for _ in range(1, self.num_of_layer+1)])

        self.mu_logvar_dev = nn.ModuleList([nn.Sequential(
            nn.Linear((self.num_of_predict - layer_idx), self.mu_dim * 2),
            nn.Linear(self.mu_dim * 2, self.mu_dim * 2)
        ) for layer_idx in range(1, self.num_of_layer+1)])

        self.mu_logvar_bias = nn.ModuleList([nn.Sequential(
            nn.Linear(self.num_of_predict, self.mu_dim * 2),
            nn.Linear(self.mu_dim * 2, self.mu_dim * 2)
        ) for _ in range(1, self.num_of_layer+1)])

    def _get_mu_logvar_dev(self, x, idx):
        out = self.mu_logvar_dev[idx](x)
        mu, log_var = torch.split(out, [self.mu_dim, self.mu_dim], dim=-1)
        return mu, log_var

    def _get_mu_logvar_bias(self, x, idx):
        out = self.mu_logvar_bias[idx](x)
        mu, log_var = torch.split(out, [self.mu_dim, self.mu_dim], dim=-1)
        return mu, log_var

    def _get_diff(self, x, n, dim):
        assert n > 0
        for i in range(n):
            x = torch.diff(x, dim=dim)
        return x

    def forward(self, batch):

        taylor_bias, taylor_dev, taylor_dev_mu, taylor_dev_logvar, taylor_bias_mu, taylor_bias_logvar \
            = [], [], [], [], [], []
        for layer_idx in range(self.num_of_layer):
            x, t, tid = batch

            x = self.data_embedding(x)

            t = self.temporal_embedding(t)
            t_his = t[:, :, :, :self.num_of_history]
            t_pred = t[:, :, :, self.num_of_history:]

            tid_pred = tid[:, :, :, self.num_of_history:]
            dtid_his = self._get_diff(tid[:, :, :, :self.num_of_history], n=layer_idx+1, dim=-1)
            dtid_pred = self._get_diff(tid[:, :, :, self.num_of_history:], n=layer_idx+1, dim=-1)

            dx_his = self._get_diff(x, n=layer_idx+1, dim=-1)
            dt_his = self._get_diff(t_his, n=layer_idx+1, dim=-1)
            dt_his = torch.where(dt_his < self.threshold, self.threshold, dt_his)
            dt_pred = self._get_diff(t_pred, n=layer_idx+1, dim=-1)

            x = dx_his / dt_his

            dev_z = self.encoder_dev[layer_idx](torch.cat((x + dt_his, dtid_his), dim=2))
            dev_z = self.transform_dev[layer_idx](dev_z, dt_his, dt_pred)
            dev_z = self.prefixer_dev[layer_idx](torch.cat((dev_z + dt_pred, dtid_pred), dim=2))
            dev = self.fc_dev[layer_idx](dev_z.permute(0, 3, 1, 2))
            dev = dev[:, :, :, -1].permute(0, 2, 1).reshape(-1, self.num_of_vertices, self.input_dim, self.num_of_predict-layer_idx-1)
            dev_mu, dev_logvar = self._get_mu_logvar_dev(dev, layer_idx)

            bias_z = self.transform_bias[layer_idx](dev_z, dt_pred, t_pred)
            bias_z = self.decoder_bias[layer_idx](torch.cat((bias_z + t_pred, tid_pred), dim=2))
            bias = self.fc_bias[layer_idx](bias_z.permute(0, 3, 1, 2))
            bias = bias[:, :, :, -1].permute(0, 2, 1).reshape(-1, self.num_of_vertices, self.input_dim, self.num_of_predict)
            bias_mu, bias_logvar = self._get_mu_logvar_bias(bias, layer_idx)

            # factorial = 1
            # for factor in range(1, self.num_of_layer + 1):
            #     factorial = factorial * factor

            # taylor_bias.append(bias / factorial)
            taylor_bias.append(bias)
            taylor_dev.append(dev)
            taylor_dev_mu.append(dev_mu)
            taylor_dev_logvar.append(dev_logvar)
            taylor_bias_mu.append(bias_mu)
            taylor_bias_logvar.append(bias_logvar)

        taylor_bias = torch.sum(torch.stack(taylor_bias), dim=0)

        return taylor_bias, taylor_dev, taylor_dev_mu, taylor_dev_logvar, taylor_bias_mu, taylor_bias_logvar

