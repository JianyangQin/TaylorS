import torch
import torch.nn as nn
from .disentangle import disentangle

class loss_function(nn.Module):
    def __init__(self, wave, level, criterion):
        super(loss_function, self).__init__()
        self.wave = wave
        self.level = level
        self.criterion = criterion

    def forward(self, true, pred):
        true_l, _ = disentangle(true.cpu().numpy(), self.wave, self.level)
        true_l = torch.from_numpy(true_l).to(true.device)
        loss = self.criterion(true, pred[0]) + self.criterion(true_l, pred[1])
        return loss