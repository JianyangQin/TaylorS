import torch
import torch.nn as nn

class loss_function(nn.Module):
    def __init__(self, criterion):
        super(loss_function, self).__init__()
        self.alpha = 0.1
        self.criterion = criterion

    def kld_loss(self, mu0, logvar0, mu1, logvar1):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + (logvar0 - logvar1) - ((mu0 - mu1) ** 2 + logvar0.exp() / logvar1.exp()), dim=-1))
        return kld_loss

    def forward(self, true, pred):
        adjust_pred, base_pred, _, dev, mu0, logvar0, mu1, logvar1 = pred

        loss_adjust = self.criterion(true, adjust_pred)
        loss_base = self.criterion(true, base_pred)
        if isinstance(dev, list):
            true_dev = true
            loss_dev = 0.
            for i in range(len(dev)):
                true_dev = torch.diff(true_dev, dim=-1)
                loss_dev = self.criterion(true_dev, dev[i])
        else:
            true_dev = torch.diff(true, dim=-1)
            loss_dev = self.criterion(true_dev, dev)

        if isinstance(mu0, list):
            loss_kld = 0.
            for i in range(len(mu0)):
                loss_kld += self.kld_loss(mu0[i], logvar0[i], mu1[i], logvar1[i])
        else:
            loss_kld = self.kld_loss(mu0, logvar0, mu1, logvar1)

        loss = (1 - self.alpha) * (loss_adjust + loss_dev) + self.alpha * (loss_base + loss_kld)
        # loss = (1 - self.alpha) * (loss_adjust + loss_dev) + self.alpha * (loss_base)

        if torch.sum(torch.isnan(loss)):
            print("loss is nan")

        return loss, loss_adjust, loss_base