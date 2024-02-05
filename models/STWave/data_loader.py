import numpy as np
import torch
from utils.base_dataloader import load_base_data, load_auxiliary_data
from utils.timefeatures import get_timestamp, convert_array2timestamp
from .disentangle import disentangle

class load_stwave_base_data(load_base_data):
    def get_train_data(self, batch):
        x, _, t = batch
        xl, xh = disentangle(x, 'coif1', 1)
        return xl.to(self.device), xh.to(self.device), t.to(self.device)

    def get_finetune_data(self, batch):
        x, _, t = batch
        xl, xh = disentangle(x, 'coif1', 1)
        return xl.to(self.device), xh.to(self.device), t.to(self.device)


class load_stwave_auxiliary_data(load_auxiliary_data):
    def get_train_data(self, batch):
        x, _, t, _ = batch
        x = self.data_scalar.inverse_transform(x).numpy()
        xl, xh = disentangle(x, 'coif1', 1)
        xl = self.data_scalar.transform(torch.from_numpy(xl))
        xh = self.data_scalar.transform(torch.from_numpy(xh))
        return xl.to(self.device), xh.to(self.device), t.to(self.device)

    def get_finetune_data(self, batch):
        x, _, t, _ = batch
        x = self.data_scalar.inverse_transform(x).numpy()
        xl, xh = disentangle(x, 'coif1', 1)
        xl = self.data_scalar.transform(torch.from_numpy(xl))
        xh = self.data_scalar.transform(torch.from_numpy(xh))
        return xl.to(self.device), xh.to(self.device), t.to(self.device)
