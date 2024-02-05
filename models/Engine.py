import torch
import torch.nn as nn
import numpy as np
import operator
from utils.metrics import MAE, RMSE, MaskedMAPE
from utils.tools import find_params, select_save_metric
import os
import shutil
import matplotlib.pyplot as plt

class Engine_Adjust(nn.Module):
    def __init__(self, cur_save_path, finetune_model, adjust_model, dataloader, start_epoch, **args):
        super(Engine_Adjust, self).__init__()

        # start up parameters
        self.device = args['device']
        self.cur_save_path = cur_save_path
        self.save_metric = args['start_up']['save_metric']

        self.finetune_name = args['start_up']['base_model_name']
        self.adjust_name = args['start_up']['adjust_model_name']
        self.finetune_model = finetune_model
        self.adjust_model = adjust_model

        # dataset parameters
        self.dataset = args['data_args']['dataset']
        self.num_of_predict = args['data_args']['num_of_predict']
        self.num_of_features = args['data_args']['num_of_features']
        self.batch_size = args['data_args']['batch_size']

        # training parameters
        self.schedulare_type = args['base_train_args']['lr_decay']

        self.start_epoch = start_epoch
        self.max_epoch = args['adjust_train_args']['max_epoch']
        self.early_stop = args['adjust_train_args']['early_stop']

        self.label_scalar = args['adjust_train_args']['label_scalar']
        self.base_scalar = args['adjust_train_args']['base_scalar']
        self.adjust_scalar = args['adjust_train_args']['adjust_scalar']
        self.test_scalar = args['adjust_train_args']['test_scalar']
        self.visual = args['adjust_train_args']['visual']

        # dataset
        self.dataloader = dataloader(self.device, **args)
        self.train_loader, self.val_loader, self.test_loader, self.data_scalar = self.dataloader.load_data()

        # model
        self.build_model(self.val_loader, self.dataloader.get_finetune_data, self.dataloader.get_adjust_data)

    def build_model(self, data_loader, get_finetune_data_func, get_adjust_data_func):
        self.finetune_model.build(data_loader, get_finetune_data_func)
        self.adjust_model.build(data_loader, get_adjust_data_func)

    def predict(self, batch):
        base_pred = self.finetune_model.model(self.dataloader.get_finetune_data(batch))
        if isinstance(base_pred, tuple) or isinstance(base_pred, list):
            base_pred = base_pred[0]
        if self.base_scalar:
            base_pred = self.data_scalar.inverse_transform(base_pred)

        residual_pred, dev, mu0, logvar0, mu1, logvar1 = self.adjust_model.model(self.dataloader.get_adjust_data(batch))
        if self.adjust_scalar:
            residual_pred = self.data_scalar.inverse_transform(residual_pred)
            dev = self.data_scalar.inverse_transform(dev)

        adjust_pred = base_pred + residual_pred

        return adjust_pred, base_pred, residual_pred, dev, mu0, logvar0, mu1, logvar1

    def visualize(self, pathname, true_y, pred_y, pred_base_y):
        if true_y.ndim > 3:
            num_of_vertices = true_y.shape[1] if true_y.shape[1] < 10 else 10
            num_of_features = true_y.shape[2]
            for i in range(num_of_vertices):
                for j in range(num_of_features):
                    filename = os.path.join(pathname, 'pred_{}point_{}feat.png'.format(i, j))
                    time_step = np.arange(0, true_y.shape[0], 1)
                    fig = plt.figure(figsize=(18, 6))
                    ax = fig.add_axes([0.15, 0.3, 0.82, 0.5])
                    ax.tick_params(labelsize=56)
                    ax.plot(time_step, true_y[:, i, j, 0], color="blue", linewidth=3, label='ground-truth')
                    ax.plot(time_step, pred_y[:, i, j, 0], color="red", linewidth=3, label='adjust prediction')
                    ax.plot(time_step, pred_base_y[:, i, j, 0], color='green', linewidth=1, label='base prediction')
                    ax.legend(fontsize=16, loc='upper right')  # 自动检测要在图例中显示的元素，并且显示
                    ax.set_xlabel('tiemslot', fontsize=64)
                    ax.set_ylabel('flow', fontsize=64)
                    plt.title('adjust', fontsize=72)
                    plt.savefig(filename, bbox_inches='tight')
                    # plt.show()
        else:
            num_of_features = true_y.shape[1]
            for i in range(num_of_features):
                filename = os.path.join(pathname, 'pred_{}feat.png'.format(i))
                time_step = np.arange(0, true_y.shape[0], 1)
                fig = plt.figure(figsize=(18, 6))
                ax = fig.add_axes([0.15, 0.3, 0.82, 0.5])
                ax.tick_params(labelsize=56)
                ax.plot(time_step, true_y[:, i, 0], color="blue", linewidth=3, label='ground-truth')
                ax.plot(time_step, pred_y[:, i, 0], color="red", linewidth=3, label='adjust prediction')
                ax.plot(time_step, pred_base_y[:, i, 0], color='green', linewidth=1, label='base prediction')
                ax.legend(fontsize=16, loc='upper right')  # 自动检测要在图例中显示的元素，并且显示
                ax.set_xlabel('tiemslot', fontsize=64)
                ax.set_ylabel('flow', fontsize=64)
                plt.title('adjust', fontsize=72)
                plt.savefig(filename, bbox_inches='tight')
                # plt.show()

    def train(self):
        txt_filename = os.path.join(self.cur_save_path, 'results.txt')
        txt = open(txt_filename, 'a')
        print('Start training model {0} on {1} dataset'.format(self.adjust_name, self.dataset), file=txt)
        txt.close()

        best_epoch = 0
        best_val_loss = np.inf
        wait_for_stop = 0

        params_filename_finetune = os.path.join(self.cur_save_path, '{}_epoch_{}.params'.format(self.finetune_name, self.start_epoch))
        params_filename_adjust = os.path.join(self.cur_save_path, '{}_epoch_{}.params'.format(self.adjust_name, self.start_epoch))

        val_loss, val_loss_adjust, val_loss_base, mae, rmse, mape = self.val()

        txt = open(txt_filename, 'a')
        if val_loss_adjust < best_val_loss:
            print('Iter {:04d} | Total Loss {:.6f} | Adjust Loss {:.6f} | Base Loss {:.6f} | MAE {:02f} | RMSE {:02f} | Save Parameter'.format(
                  self.start_epoch, val_loss, val_loss_adjust, val_loss_base, mae, rmse))
            print('Iter {:04d} | Total Loss {:.6f} | Adjust Loss {:.6f} | Base Loss {:.6f} | MAE {:02f} | RMSE {:02f} | Save Parameter'.format(
                  self.start_epoch, val_loss, val_loss_adjust, val_loss_base, mae, rmse), file=txt)
            wait_for_stop = 0
            best_epoch = self.start_epoch
            best_val_loss = val_loss_adjust
            torch.save(self.finetune_model.model.state_dict(), params_filename_finetune)
            torch.save(self.adjust_model.model.state_dict(), params_filename_adjust)
        else:
            print('Iter {:04d} | Total Loss {:.6f} | Adjust Loss {:.6f} | Base Loss {:.6f} | MAE {:02f} | RMSE {:.6f}'.format(
                  self.start_epoch, val_loss, val_loss_adjust, val_loss_base, mae, rmse))
            print('Iter {:04d} | Total Loss {:.6f} | Adjust Loss {:.6f} | Base Loss {:.6f} | MAE {:02f} | RMSE {:.6f}'.format(
                  self.start_epoch, val_loss, val_loss_adjust, val_loss_base, mae, rmse), file=txt)
            wait_for_stop += 1
            if (self.early_stop is not False) and (wait_for_stop > self.early_stop):
                print('Early Stopping, best epoch: {:04d}'.format(best_epoch))
                print('Early Stopping, best epoch: {:04d}'.format(best_epoch), file=txt)
        txt.close()

        for epoch in range(self.start_epoch + 1, self.max_epoch):
            params_filename_finetune = os.path.join(self.cur_save_path, '{}_epoch_{}.params'.format(self.finetune_name, epoch))
            params_filename_adjust = os.path.join(self.cur_save_path, '{}_epoch_{}.params'.format(self.adjust_name, epoch))

            # train
            self.finetune_model.model.train(True)
            self.adjust_model.model.train(True)
            train_loss = []

            for i, batch in enumerate(self.train_loader):
                #-------------- model --------------#
                pred_y = self.predict(batch)

                # -------------- optimizer --------------#
                self.finetune_model.optimizer.zero_grad()
                self.adjust_model.optimizer.zero_grad()

                y = self.dataloader.get_ground_truth(batch)
                if self.label_scalar:
                    y = self.data_scalar.inverse_transform(y)

                loss, _, _ = self.adjust_model.criterion(y, pred_y)
                loss.backward()

                if self.finetune_model.clip is not False:
                    torch.nn.utils.clip_grad_norm_(parameters=self.finetune_model.model.parameters(),
                                                   max_norm=self.finetune_model.clip,
                                                   norm_type=2)
                if self.adjust_model.clip is not False:
                    torch.nn.utils.clip_grad_norm_(parameters=self.adjust_model.model.parameters(),
                                                   max_norm=self.adjust_model.clip,
                                                   norm_type=2)
                self.finetune_model.optimizer.step()
                self.adjust_model.optimizer.step()

                train_loss.append(loss.item())

            val_loss, val_loss_adjust, val_loss_base, mae, rmse, mape = self.val()
            cur_loss = select_save_metric(self.save_metric, val_loss, mae, rmse, mape)

            txt = open(txt_filename, 'a')
            if val_loss_adjust < best_val_loss:
                print('Iter {:04d} | Total Loss {:.6f} | Adjust Loss {:.6f} | Base Loss {:.6f} | MAE {:02f} | RMSE {:02f} | Save Parameter'.format(
                      epoch, val_loss, val_loss_adjust, val_loss_base, mae, rmse))
                print('Iter {:04d} | Total Loss {:.6f} | Adjust Loss {:.6f} | Base Loss {:.6f} | MAE {:02f} | RMSE {:02f} | Save Parameter'.format(
                      epoch, val_loss, val_loss_adjust, val_loss_base, mae, rmse), file=txt)
                wait_for_stop = 0
                best_epoch = epoch
                best_val_loss = val_loss_adjust
                torch.save(self.finetune_model.model.state_dict(), params_filename_finetune)
                torch.save(self.adjust_model.model.state_dict(), params_filename_adjust)
            else:
                print('Iter {:04d} | Total Loss {:.6f} | Adjust Loss {:.6f} | Base Loss {:.6f} | MAE {:02f} | RMSE {:.6f}'.format(
                      epoch, val_loss, val_loss_adjust, val_loss_base, mae, rmse))
                print('Iter {:04d} | Total Loss {:.6f} | Adjust Loss {:.6f} | Base Loss {:.6f} | MAE {:02f} | RMSE {:.6f}'.format(
                      epoch, val_loss, val_loss_adjust, val_loss_base, mae, rmse), file=txt)
                wait_for_stop += 1
                if (self.early_stop is not False) and (wait_for_stop > self.early_stop):
                    print('Early Stopping, best epoch: {:04d}'.format(best_epoch))
                    print('Early Stopping, best epoch: {:04d}'.format(best_epoch), file=txt)
                    break
            txt.close()

            if self.finetune_model.scheduler is not None:
                if self.schedulare_type == 'ReduceLROnPlateau':
                    self.finetune_model.scheduler.step(cur_loss)
                else:
                    self.finetune_model.scheduler.step()

            if self.adjust_model.scheduler is not None:
                self.adjust_model.scheduler.step()

        best_epoch_filename_finetune = os.path.join(self.cur_save_path, '{}_epoch_{}.params'.format(self.finetune_name, best_epoch))
        best_epoch_filename_adjust = os.path.join(self.cur_save_path, '{}_epoch_{}.params'.format(self.adjust_name, best_epoch))

        best_param_filename_finetune = os.path.join(self.cur_save_path, '{}_best_epoch_{}.params'.format(self.finetune_name, best_epoch))
        best_param_filename_adjust = os.path.join(self.cur_save_path, '{}_best_epoch_{}.params'.format(self.adjust_name, best_epoch))

        if os.path.exists(best_epoch_filename_finetune):
            shutil.copy(best_epoch_filename_finetune, best_param_filename_finetune)
        if os.path.exists(best_epoch_filename_adjust):
            shutil.copy(best_epoch_filename_adjust, best_param_filename_adjust)

        return best_epoch

    def val(self):
        self.finetune_model.model.train(False)
        self.adjust_model.model.train(False)

        curv_true_y, curv_pred_y = [], []
        with torch.no_grad():
            val_loss, val_loss_adjust, val_loss_base = [], [], []

            for i, batch in enumerate(self.val_loader):
                # -------------- val --------------#
                adjust_pred, base_pred, residual_pred, dev_pred, mu1, var1, mu2, var2 = self.predict(batch)

                # -------------- loss --------------#
                y = self.dataloader.get_ground_truth(batch)
                if self.label_scalar or self.test_scalar:
                    y = self.data_scalar.inverse_transform(y)
                if self.base_scalar is False and self.adjust_scalar is False and self.test_scalar:
                    adjust_pred = self.data_scalar.inverse_transform(adjust_pred)
                    dev_pred = self.data_scalar.inverse_transform(dev_pred)
                if self.base_scalar is False and self.test_scalar:
                    base_pred = self.data_scalar.inverse_transform(base_pred)
                y_pred = (adjust_pred, base_pred, residual_pred, dev_pred, mu1, var1, mu2, var2)

                loss, loss_adjust, loss_base = self.adjust_model.criterion(y, y_pred)
                val_loss.append(loss.item())
                val_loss_adjust.append(loss_adjust.item())
                val_loss_base.append((loss_base.item()))

                curv_true_y.append(y.detach().cpu().numpy())
                if isinstance(y_pred, tuple) or isinstance(y_pred, list):
                    curv_pred_y.append(y_pred[0].detach().cpu().numpy())
                else:
                    curv_pred_y.append(y_pred.detach().cpu().numpy())

            val_loss = sum(val_loss) / len(val_loss)
            val_loss_adjust = sum(val_loss_adjust) / len(val_loss_adjust)
            val_loss_base = sum(val_loss_base) / len(val_loss_base)

            curv_true_y = np.concatenate(curv_true_y, 0)
            curv_pred_y = np.concatenate(curv_pred_y, 0)

            mae = MAE(curv_true_y.reshape(-1, 1), curv_pred_y.reshape(-1, 1))
            rmse = RMSE(curv_true_y.reshape(-1, 1), curv_pred_y.reshape(-1, 1))
            mape = MaskedMAPE(curv_true_y.reshape(-1, 1), curv_pred_y.reshape(-1, 1), 0)

        return val_loss, val_loss_adjust, val_loss_base, mae, rmse, mape

    def test(self):
        best_param_finetune = find_params(self.cur_save_path, '{}_best'.format(self.finetune_name))
        best_param_adjust = find_params(self.cur_save_path, '{}_best'.format(self.adjust_name))

        if (best_param_finetune is not None) and (best_param_adjust is not None):
            txt_filename = os.path.join(self.cur_save_path, 'results.txt')

            self.finetune_model.model.load_state_dict(torch.load(best_param_finetune))
            self.adjust_model.model.load_state_dict(torch.load(best_param_adjust))

            self.finetune_model.model.train(False)
            self.adjust_model.model.train(False)
            with torch.no_grad():
                curv_adjust_pred, curv_base_pred, curv_true_y = [], [], []

                for i, batch in enumerate(self.test_loader):
                    # -------------- test --------------#
                    adjust_pred, base_pred, residual_pred, _, _, _, _, _ = self.predict(batch)

                    y = self.dataloader.get_ground_truth(batch)

                    if self.label_scalar or self.test_scalar:
                        y = self.data_scalar.inverse_transform(y.detach().cpu())
                    if self.base_scalar is False and self.adjust_scalar is False and self.test_scalar:
                        adjust_pred = self.data_scalar.inverse_transform(adjust_pred.detach().cpu())
                    if self.base_scalar is False and self.test_scalar:
                        base_pred = self.data_scalar.inverse_transform(base_pred.detach().cpu())

                    curv_true_y.append(y.detach().cpu().numpy())
                    curv_adjust_pred.append(adjust_pred.detach().cpu().numpy())
                    curv_base_pred.append(base_pred.detach().cpu().numpy())

                curv_true_y = np.concatenate(curv_true_y, 0)
                curv_adjust_pred = np.concatenate(curv_adjust_pred, 0)
                curv_base_pred = np.concatenate(curv_base_pred, 0)

                if self.visual:
                    visual_path = os.path.join(self.cur_save_path, 'figs')
                    os.makedirs(visual_path, exist_ok=True)
                    visual_data_path = os.path.join(visual_path, 'test')
                    os.makedirs(visual_data_path, exist_ok=True)
                    self.visualize(visual_data_path, curv_true_y, curv_adjust_pred, curv_base_pred)

                txt = open(txt_filename, 'a')
                # finetune results
                # one point result
                f_mae, f_rmse, f_mape = [], [], []
                print('\n\n--------------------------Finetune Results--------------------------')
                print('\n\n--------------------------Finetune Results--------------------------', file=txt)
                for i in range(self.num_of_predict):
                    mae = MAE(curv_true_y[..., i], curv_base_pred[..., i])
                    rmse = RMSE(curv_true_y[..., i], curv_base_pred[..., i])
                    mape = MaskedMAPE(curv_true_y[..., i], curv_base_pred[..., i], 0)
                    f_mae.append(mae)
                    f_rmse.append(rmse)
                    f_mape.append(mape)
                    print('Result of {:04d} points: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(i, mae, rmse, mape))
                    print('Result of {:04d} points: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(i, mae, rmse, mape),
                          file=txt)

                # overall results
                mae = MAE(curv_true_y.reshape(-1, 1), curv_base_pred.reshape(-1, 1))
                rmse = RMSE(curv_true_y.reshape(-1, 1), curv_base_pred.reshape(-1, 1))
                mape = MaskedMAPE(curv_true_y.reshape(-1, 1), curv_base_pred.reshape(-1, 1), 0)
                f_mae.append(mae)
                f_rmse.append(rmse)
                f_mape.append(mape)
                print('Overall Results: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(mae, rmse, mape))
                print('Overall Results: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(mae, rmse, mape), file=txt)


                # Adjust results
                c2f_mae, c2f_rmse, c2f_mape = [], [], []
                # one point result
                print('\n\n--------------------------Adjust Results--------------------------')
                print('\n\n--------------------------Adjust Results--------------------------', file=txt)
                for i in range(self.num_of_predict):
                    mae = MAE(curv_true_y[..., i], curv_adjust_pred[..., i])
                    rmse = RMSE(curv_true_y[..., i], curv_adjust_pred[..., i])
                    mape = MaskedMAPE(curv_true_y[..., i], curv_adjust_pred[..., i], 0)
                    c2f_mae.append(mae)
                    c2f_rmse.append(rmse)
                    c2f_mape.append(mape)
                    print('Result of {:04d} points: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(i, mae, rmse, mape))
                    print('Result of {:04d} points: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(i, mae, rmse, mape), file=txt)

                # overall results
                mae = MAE(curv_true_y.reshape(-1, 1), curv_adjust_pred.reshape(-1, 1))
                rmse = RMSE(curv_true_y.reshape(-1, 1), curv_adjust_pred.reshape(-1, 1))
                mape = MaskedMAPE(curv_true_y.reshape(-1, 1), curv_adjust_pred.reshape(-1, 1), 0)
                c2f_mae.append(mae)
                c2f_rmse.append(rmse)
                c2f_mape.append(mape)
                print('Overall Results: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(mae, rmse, mape))
                print('Overall Results: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(mae, rmse, mape), file=txt)
                txt.close()

            return c2f_mae, c2f_rmse, c2f_mape, f_mae, f_rmse, f_mape
        else:
            raise ValueError('params file does not exist!')


class Engine_Train(nn.Module):
    def __init__(self, cur_save_path, train_model, dataloader, start_epoch, **args):
        super(Engine_Train, self).__init__()
        self.device = args['device']

        # start up parameters
        self.cur_save_path = cur_save_path
        self.save_metric = args['start_up']['save_metric']
        self.train_name = args['start_up']['base_model_name']
        self.train_model = train_model

        # dataset parameters
        self.dataset = args['data_args']['dataset']
        self.num_of_predict = args['data_args']['num_of_predict']
        self.batch_size = args['data_args']['batch_size']

        # training parameters
        self.start_epoch = start_epoch
        self.max_epoch = args['base_train_args']['max_epoch']
        self.early_stop = args['base_train_args']['early_stop']
        self.schedulare_type = args['base_train_args']['lr_decay']

        self.label_scalar = args['base_train_args']['label_scalar']
        self.pred_scalar = args['base_train_args']['pred_scalar']
        self.test_scalar = args['base_train_args']['test_scalar']
        self.visual = args['base_train_args']['visual']

        # dataset
        self.dataloader = dataloader(self.device, **args)
        self.train_loader, self.val_loader, self.test_loader, self.data_scalar = self.dataloader.load_data()

        # model
        self.build_model(self.val_loader, self.dataloader.get_train_data)

    def build_model(self, data_loader, get_train_data_func):
        self.train_model.build(data_loader, get_train_data_func)

    def predict(self, batch):
        pred = self.train_model.model(self.dataloader.get_train_data(batch))
        if self.pred_scalar:
            pred = self.data_scalar.inverse_transform(pred)
        return pred

    def visualize(self, pathname, true_y, pred_y):
        if true_y.ndim > 3:
            num_of_vertices = true_y.shape[1] if true_y.shape[1] < 10 else 10
            num_of_features = true_y.shape[2]
            for i in range(num_of_vertices):
                for j in range(num_of_features):
                    filename = os.path.join(pathname, 'pred_{}point_{}feat.png'.format(i, j))
                    time_step = np.arange(0, true_y.shape[0], 1)
                    fig = plt.figure(figsize=(18, 6))
                    ax = fig.add_axes([0.15, 0.3, 0.82, 0.5])
                    ax.tick_params(labelsize=56)
                    ax.plot(time_step, true_y[:, i, j, 0], color="blue", linewidth=3, label='ground-truth')
                    ax.plot(time_step, pred_y[:, i, j, 0], color="red", linewidth=3, label='prediction')
                    ax.legend(fontsize=16, loc='upper right')  # 自动检测要在图例中显示的元素，并且显示
                    ax.set_xlabel('tiemslot', fontsize=64)
                    ax.set_ylabel('flow', fontsize=64)
                    plt.title('Base', fontsize=72)
                    plt.savefig(filename, bbox_inches='tight')
                    # plt.show()
        else:
            num_of_features = true_y.shape[1]
            for i in range(num_of_features):
                filename = os.path.join(pathname, 'pred_{}feat.png'.format(i))
                time_step = np.arange(0, true_y.shape[0], 1)
                fig = plt.figure(figsize=(18, 6))
                ax = fig.add_axes([0.15, 0.3, 0.82, 0.5])
                ax.tick_params(labelsize=56)
                ax.plot(time_step, true_y[:, i, 0], color="blue", linewidth=3, label='ground-truth')
                ax.plot(time_step, pred_y[:, i, 0], color="red", linewidth=3, label='prediction')
                ax.legend(fontsize=16, loc='upper right')  # 自动检测要在图例中显示的元素，并且显示
                ax.set_xlabel('tiemslot', fontsize=64)
                ax.set_ylabel('flow', fontsize=64)
                plt.title('Base', fontsize=72)
                plt.savefig(filename, bbox_inches='tight')
                # plt.show()

    def train(self):
        txt_filename = os.path.join(self.cur_save_path, 'results.txt')
        txt = open(txt_filename, 'a')
        print('Start training model {0} on {1} dataset'.format(self.train_name, self.dataset), file=txt)
        txt.close()

        best_epoch = 0
        best_val_loss = np.inf
        wait_for_stop = 0

        params_filename_train = os.path.join(self.cur_save_path, '{}_epoch_{}.params'.format(self.train_name, self.start_epoch))

        val_loss, mae, rmse, mape = self.val()
        cur_loss = select_save_metric(self.save_metric, val_loss, mae, rmse, mape)

        txt = open(txt_filename, 'a')
        if cur_loss < best_val_loss:
            print('Iter {:04d} | Total Loss {:.6f} | MAE {:.2f} | RMSE {:.2f} | Save Parameter'.format(self.start_epoch, val_loss, mae, rmse))
            print('Iter {:04d} | Total Loss {:.6f} | MAE {:.2f} | RMSE {:.2f} | Save Parameter'.format(self.start_epoch, val_loss, mae, rmse), file=txt)
            wait_for_stop = 0
            best_epoch = self.start_epoch
            best_val_loss = cur_loss
            torch.save(self.train_model.model.state_dict(), params_filename_train)
        else:
            print('Iter {:04d} | Total Loss {:.6f} | MAE {:.2f} | RMSE {:.2f} '.format(self.start_epoch, val_loss, mae, rmse))
            print('Iter {:04d} | Total Loss {:.6f} | MAE {:.2f} | RMSE {:.2f} '.format(self.start_epoch, val_loss, mae, rmse), file=txt)
            wait_for_stop += 1
            if (self.early_stop is not False) and (wait_for_stop > self.early_stop):
                print('Early Stopping, best epoch: {:04d}'.format(best_epoch))
                print('Early Stopping, best epoch: {:04d}'.format(best_epoch), file=txt)
        txt.close()

        for epoch in range(self.start_epoch + 1, self.max_epoch):
            params_filename_train = os.path.join(self.cur_save_path, '{}_epoch_{}.params'.format(self.train_name, epoch))

            # train
            self.train_model.model.train(True)
            train_loss = []

            for i, batch in enumerate(self.train_loader):
                #-------------- model --------------#
                y_pred = self.predict(batch)

                # -------------- optimizer --------------#
                self.train_model.optimizer.zero_grad()

                y = self.dataloader.get_ground_truth(batch)
                if self.label_scalar:
                    y = self.data_scalar.inverse_transform(y)

                loss = self.train_model.criterion(y, y_pred)
                loss.backward()

                if self.train_model.clip is not False:
                    torch.nn.utils.clip_grad_norm_(parameters=self.train_model.model.parameters(),
                                                   max_norm=self.train_model.clip,
                                                   norm_type=2)

                self.train_model.optimizer.step()

                train_loss.append(loss.item())

            val_loss, mae, rmse, mape = self.val()
            cur_loss = select_save_metric(self.save_metric, val_loss, mae, rmse, mape)

            txt = open(txt_filename, 'a')
            if cur_loss < best_val_loss:
                print('Iter {:04d} | Total Loss {:.6f} | MAE {:.2f} | RMSE {:.2f} | Save Parameter'.format(epoch, val_loss, mae, rmse))
                print('Iter {:04d} | Total Loss {:.6f} | MAE {:.2f} | RMSE {:.2f} | Save Parameter'.format(epoch, val_loss, mae, rmse), file=txt)
                wait_for_stop = 0
                best_epoch = epoch
                best_val_loss = cur_loss
                torch.save(self.train_model.model.state_dict(), params_filename_train)
            else:
                print('Iter {:04d} | Total Loss {:.6f} | MAE {:.2f} | RMSE {:.2f} '.format(epoch, val_loss, mae, rmse))
                print('Iter {:04d} | Total Loss {:.6f} | MAE {:.2f} | RMSE {:.2f} '.format(epoch, val_loss, mae, rmse), file=txt)
                wait_for_stop += 1
                if (self.early_stop is not False) and (wait_for_stop > self.early_stop):
                    print('Early Stopping, best epoch: {:04d}'.format(best_epoch))
                    print('Early Stopping, best epoch: {:04d}'.format(best_epoch), file=txt)
                    break
            txt.close()

            if self.train_model.scheduler is not None:
                if self.schedulare_type ==  'ReduceLROnPlateau':
                    self.train_model.scheduler.step(cur_loss)
                else:
                    self.train_model.scheduler.step()

        best_epoch_filename_train = os.path.join(self.cur_save_path, '{}_epoch_{}.params'.format(self.train_name, best_epoch))
        best_param_filename_train = os.path.join(self.cur_save_path, '{}_best_epoch_{}.params'.format(self.train_name, best_epoch))
        if os.path.exists(best_epoch_filename_train):
            shutil.copy(best_epoch_filename_train, best_param_filename_train)

        return best_epoch, best_param_filename_train

    def val(self):
        self.train_model.model.train(False)

        curv_true_y, curv_pred_y = [], []
        with torch.no_grad():
            val_loss = []

            for i, batch in enumerate(self.val_loader):
                # -------------- val --------------#
                y_pred = self.predict(batch)

                # -------------- loss --------------#
                y = self.dataloader.get_ground_truth(batch)
                if self.label_scalar or self.test_scalar:
                    y = self.data_scalar.inverse_transform(y)
                if self.pred_scalar is False and self.test_scalar:
                    y_pred = self.data_scalar.inverse_transform(y_pred)

                loss = self.train_model.criterion(y, y_pred)
                val_loss.append(loss.item())

                curv_true_y.append(y.detach().cpu().numpy())
                if isinstance(y_pred, tuple) or isinstance(y_pred, list):
                    curv_pred_y.append(y_pred[0].detach().cpu().numpy())
                else:
                    curv_pred_y.append(y_pred.detach().cpu().numpy())

            val_loss = sum(val_loss) / len(val_loss)

            curv_true_y = np.concatenate(curv_true_y, 0)
            curv_pred_y = np.concatenate(curv_pred_y, 0)

            mae = MAE(curv_true_y.reshape(-1, 1), curv_pred_y.reshape(-1, 1))
            rmse = RMSE(curv_true_y.reshape(-1, 1), curv_pred_y.reshape(-1, 1))
            mape = MaskedMAPE(curv_true_y.reshape(-1, 1), curv_pred_y.reshape(-1, 1), 0)

        return val_loss, mae, rmse, mape

    def test(self):
        best_param_train = find_params(self.cur_save_path, 'best')

        if best_param_train is not None:
            txt_filename = os.path.join(self.cur_save_path, 'results.txt')

            self.train_model.model.load_state_dict(torch.load(best_param_train))

            self.train_model.model.train(False)
            with torch.no_grad():
                curv_true_y, curv_pred_y = [], []

                for i, batch in enumerate(self.test_loader):
                    # -------------- test --------------#
                    y_pred = self.predict(batch)

                    y = self.dataloader.get_ground_truth(batch)

                    if self.label_scalar or self.test_scalar:
                        y = self.data_scalar.inverse_transform(y)
                    if self.pred_scalar is False and self.test_scalar:
                        y_pred = self.data_scalar.inverse_transform(y_pred)

                    curv_true_y.append(y.detach().cpu().numpy())
                    if isinstance(y_pred, list):
                        curv_pred_y.append(y_pred[0].detach().cpu().numpy())
                    else:
                        curv_pred_y.append(y_pred.detach().cpu().numpy())

                curv_true_y = np.concatenate(curv_true_y, 0)
                curv_pred_y = np.concatenate(curv_pred_y, 0)

                if self.visual:
                    visual_path = os.path.join(self.cur_save_path, 'figs')
                    os.makedirs(visual_path, exist_ok=True)
                    visual_data_path = os.path.join(visual_path, 'test')
                    os.makedirs(visual_data_path, exist_ok=True)
                    self.visualize(visual_data_path, curv_true_y, curv_pred_y)

                txt = open(txt_filename, 'a')
                maes, rmses, mapes = [], [], []
                # one point result
                print('--------------------------Train Results--------------------------')
                print('--------------------------Train Results--------------------------', file=txt)
                for i in range(self.num_of_predict):
                    mae = MAE(curv_true_y[..., i], curv_pred_y[..., i])
                    rmse = RMSE(curv_true_y[..., i], curv_pred_y[..., i])
                    mape = MaskedMAPE(curv_true_y[..., i], curv_pred_y[..., i], 0)
                    maes.append(mae)
                    rmses.append(rmse)
                    mapes.append(mape)
                    print('Result of {:04d} points: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(i, mae, rmse, mape))
                    print('Result of {:04d} points: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(i, mae, rmse, mape), file=txt)

                # overall results
                mae = MAE(curv_true_y.reshape(-1, 1), curv_pred_y.reshape(-1, 1))
                rmse = RMSE(curv_true_y.reshape(-1, 1), curv_pred_y.reshape(-1, 1))
                mape = MaskedMAPE(curv_true_y.reshape(-1, 1), curv_pred_y.reshape(-1, 1), 0)
                maes.append(mae)
                rmses.append(rmse)
                mapes.append(mape)
                print('Overall Results: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(mae, rmse, mape))
                print('Overall Results: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(mae, rmse, mape), file=txt)

            return maes, rmses, mapes
        else:
            raise ValueError('params file does not exist!')

