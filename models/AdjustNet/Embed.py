import torch
import torch.nn as nn
import torch.nn.functional as F
import re

class Conv2D(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(Conv2D, self).__init__()
        self.activation = activation

        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x.permute(0, 2, 1, 3)

class DataEmbedding(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(DataEmbedding, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([Conv2D(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            use_bias=use_bias, activation=activation, bn_decay=bn_decay)
            for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

class DiscreteTemporalEmbedding(nn.Module):
    def __init__(self, features, channel, num_of_vertices,
                 time_per_day, day_per_week=7, day_per_month=31, month_per_year=12):
        super(DiscreteTemporalEmbedding, self).__init__()
        self.features = features
        self.time_per_day = time_per_day
        self.day_per_week = day_per_week
        self.day_per_month = day_per_month
        self.month_per_year = month_per_year
        self.num_of_vertices = num_of_vertices
        if self.features == 2:
            # self.fc = DataEmbedding(input_dims=[time_per_day + day_per_week, channel],
            #                         units=[channel, channel],
            #                         activations=[F.relu, None],
            #                         bn_decay=0.01)
            self.fc = nn.Linear(time_per_day + day_per_week, channel, bias=False)
        elif self.features == 4:
            # self.fc = DataEmbedding(input_dims=[time_per_day + day_per_week + day_per_month + month_per_year, channel],
            #                         units=[channel, channel],
            #                         activations=[F.relu, None],
            #                         bn_decay=0.01)
            self.fc = nn.Linear(time_per_day + day_per_week + day_per_month + month_per_year, channel, bias=False)

    def forward(self, t):
        if self.features == 2:
            dayofweek = torch.empty(t.shape[0], t.shape[1], self.day_per_week).to(t.device)
            timeofday = torch.empty(t.shape[0], t.shape[1], self.time_per_day).to(t.device)
            for i in range(t.shape[0]):
                dayofweek[i] = F.one_hot(t[..., 0][i].to(torch.int64) % self.day_per_week, self.day_per_week)
            for j in range(t.shape[0]):
                timeofday[j] = F.one_hot(t[..., 1][j].to(torch.int64) % self.time_per_day, self.time_per_day)
            t = torch.cat((dayofweek, timeofday), dim=-1)
            t = self.fc(t).permute(0, 2, 1)
            t = t.unsqueeze(dim=1)
            t = t.repeat(1, self.num_of_vertices, 1, 1)
        elif self.features == 4:
            monthofyear = torch.empty(t.shape[0], t.shape[1], self.month_per_year).to(t.device)
            dayofmonth = torch.empty(t.shape[0], t.shape[1], self.day_per_month).to(t.device)
            dayofweek = torch.empty(t.shape[0], t.shape[1], self.day_per_week).to(t.device)
            timeofday = torch.empty(t.shape[0], t.shape[1], self.time_per_day).to(t.device)
            for i in range(t.shape[0]):
                monthofyear[i] = F.one_hot(t[..., 0][i].to(torch.int64) % self.month_per_year, self.month_per_year)
            for j in range(t.shape[0]):
                dayofmonth[j] = F.one_hot(t[..., 1][j].to(torch.int64) % self.day_per_month, self.day_per_month)
            for k in range(t.shape[0]):
                dayofweek[k] = F.one_hot(t[..., 2][k].to(torch.int64) % self.day_per_week, self.day_per_week)
            for m in range(t.shape[0]):
                timeofday[m] = F.one_hot(t[..., 3][m].to(torch.int64) % self.time_per_day, self.time_per_day)
            t = torch.cat((monthofyear, dayofmonth, dayofweek, timeofday), dim=-1)
            t = self.fc(t).permute(0, 2, 1)
            t = t.unsqueeze(dim=1)
            t = t.repeat(1, self.num_of_vertices, 1, 1)
        return t

class ContinousTemporalEmbedding(nn.Module):
    def __init__(self, num_of_vertices, channel, freq):
        super(ContinousTemporalEmbedding, self).__init__()

        freq = re.split(r'(\d+)', freq)[-1].lower()

        self.num_of_vertices = num_of_vertices
        freq_map = {'h': 4, 't': 5, 'min':5, 's': 6, 'm': 1,
                    'a': 1, 'w': 2, 'd': 3, 'b': 3}
        dims = freq_map[freq]
        self.embed = nn.Linear(dims, channel, bias=False)

    def forward(self, t):
        t = self.embed(t)
        t = t.unsqueeze(dim=1)
        t = t.repeat(1, self.num_of_vertices, 1, 1)
        return t.permute(0, 1, 3, 2)