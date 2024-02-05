from utils.base_dataloader import load_base_data, load_auxiliary_data

class load_gman_base_data(load_base_data):
    def get_train_data(self, batch):
        x, _, t = batch
        return x.to(self.device), t.to(self.device)

    def get_finetune_data(self, batch):
        x, _, t = batch
        return x.to(self.device), t.to(self.device)

class load_gman_auxiliary_data(load_auxiliary_data):
    def get_train_data(self, batch):
        x, _, t, _ = batch
        return x.to(self.device), t.to(self.device)

    def get_finetune_data(self, batch):
        x, _, t, _ = batch
        return x.to(self.device), t.to(self.device)