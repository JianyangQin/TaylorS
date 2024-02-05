import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import configparser
import yaml
from models.ModelsBuilder import models_builder, adjust_builder
from models.Engine import Engine_Train, Engine_Adjust
from models.DataLoader import data_loader
from utils.tools import find_params, find_epoch
from datetime import datetime
import random
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def set_seed(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/HZ-METRO/HZMETRO_d2stgnn.yaml', type=str,
                        help="configuration file path")
    # parser.add_argument("--config", default='configs/HZ-METRO/HZMETRO_stwave.yaml', type=str,
    #                     help="configuration file path")
    # parser.add_argument("--config", default='configs/HZ-METRO/HZMETRO_stgode.yaml', type=str,
    #                     help="configuration file path")
    # parser.add_argument("--config", default='configs/HZ-METRO/HZMETRO_stgncde.yaml', type=str,
    #                     help="configuration file path")
    # parser.add_argument("--config", default='configs/SH-METRO/SHMETRO_d2stgnn.yaml', type=str,
    #                     help="configuration file path")
    # parser.add_argument("--config", default='configs/PEMS08/PEMS08_d2stgnn.yaml', type=str,
    #                     help="configuration file path")
    # parser.add_argument("--config", default='configs/PEMS04/PEMS04_d2stgnn.yaml', type=str,
    #                     help="configuration file path")
    # parser.add_argument("--config", default='configs/Wind/Wind_d2stgnn.yaml', type=str,
    #                     help="configuration file path")
    # parser.add_argument("--config", default='configs/PEMS08/PEMS08_stwave.yaml', type=str,
    #                     help="configuration file path")

    parser.add_argument("--gpu", default='6', type=str,
                        help="gpu device")
    parser.add_argument('--use_multi_gpu', default=False, action='store_true',
                        help='use multiple gpus')
    parser.add_argument("--backbone", default='gwnet', type=str,
                        help="backbone of adjustnet")
    parser.add_argument("--test", default=False, action='store_true')
    parser.add_argument("--test_path", default='', type=str)
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_path", default='', type=str)
    args = parser.parse_args()

    print('Read configuration file: %s' % (args.config))
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    configs['use_multi_gpu'] = args.use_multi_gpu
    if args.use_multi_gpu:
        configs['devices'] = args.gpu.replace(' ', '')
        device_ids = args.gpu.split(',')
        configs['device_ids'] = list(range(len(device_ids)))
        configs['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        configs['device_ids'] = None
        configs['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    save_path = configs['start_up']['save_path']
    base_name = configs['start_up']['base_model_name']
    adjust_name = configs['start_up']['adjust_model_name']
    dataset_name = configs['data_args']['dataset']
    num_of_predict = configs['data_args']['num_of_predict']

    is_skip_train = False
    is_skip_adjust = False
    train_start_epoch = 0
    adjust_start_epoch = 0
    train_checkpoint = None
    finetune_checkpoint = None
    adjust_checkpoint = None

    if args.resume:
        if args.resume_path == '':
            cur_save_list = os.listdir(save_path)
            cur_save_list.sort(key=lambda fn: os.path.getmtime(save_path + "/" + fn))
            cur_save_path = os.path.join(save_path, cur_save_list[-1])
            cur_subpath_list = os.listdir(cur_save_path)
        else:
            if os.path.exists(args.resume_path):
                cur_save_path = args.resume_path
                cur_subpath_list = os.listdir(cur_save_path)
            else:
                raise ValueError('resume path does not exist')

        if 'results.txt' in cur_subpath_list:
            # train and guid has finished
            is_skip_train = True
            is_skip_adjust = True
            # find best train model
            cur_train_save_path = os.path.join(cur_save_path, base_name)
            train_checkpoint = find_params(cur_train_save_path, 'best')
            train_start_epoch = configs['base_train_args']['max_epoch']
            # find best adjust model
            cur_adjust_save_path = os.path.join(cur_save_path, adjust_name)
            finetune_checkpoint = find_params(cur_adjust_save_path, '{}_best'.format(base_name))
            adjust_checkpoint = find_params(cur_adjust_save_path, '{}_best'.format(adjust_name))
            adjust_start_epoch = configs['adjust_train_args']['max_epoch']
        elif adjust_name in cur_subpath_list:
            # resume from adjust
            is_skip_train = True
            is_skip_adjust = False
            # find best train model
            cur_train_save_path = os.path.join(cur_save_path, base_name)
            train_checkpoint = find_params(cur_train_save_path, 'best')
            train_start_epoch = configs['base_train_args']['max_epoch']
            # find resume epoch of adjust model
            cur_adjust_save_path = os.path.join(cur_save_path, adjust_name)
            finetune_checkpoint = find_params(cur_adjust_save_path, '{}_'.format(base_name))
            adjust_checkpoint = find_params(cur_adjust_save_path, '{}_'.format(adjust_name))
            adjust_start_epoch = find_epoch(adjust_checkpoint)
        elif base_name in cur_subpath_list:
            # resume from train
            is_skip_train = False
            is_skip_adjust = False
            # find resume epoch of train model
            cur_train_save_path = os.path.join(cur_save_path, base_name)
            train_checkpoint = find_params(cur_train_save_path)
            train_start_epoch = find_epoch(train_checkpoint)
        else:
            raise ValueError("resume error: model does not exist")
    elif args.test:
        if args.test_path == '':
            cur_save_list = os.listdir(save_path)
            cur_save_list.sort(key=lambda fn: os.path.getmtime(save_path + "/" + fn))
            cur_save_path = os.path.join(save_path, cur_save_list[-1])
            cur_subpath_list = os.listdir(cur_save_path)
        else:
            if os.path.exists(args.test_path):
                cur_save_path = args.test_path
                cur_subpath_list = os.listdir(cur_save_path)
            else:
                raise ValueError('test path does not exist')

        if (adjust_name in cur_subpath_list) and (base_name in cur_subpath_list):
            # test for both train and adjust model
            is_skip_train = True
            is_skip_adjust = True

            # find best train model
            cur_train_save_path = os.path.join(cur_save_path, base_name)
            train_checkpoint = find_params(cur_train_save_path, 'best')
            train_start_epoch = configs['base_train_args']['max_epoch']

            # find best adjust model
            cur_adjust_save_path = os.path.join(cur_save_path, adjust_name)
            finetune_checkpoint = find_params(cur_adjust_save_path, '{}_best'.format(base_name))
            adjust_checkpoint = find_params(cur_adjust_save_path, '{}_best'.format(adjust_name))
            adjust_start_epoch = configs['adjust_train_args']['max_epoch']
        else:
            raise ValueError("test error: train or adjust model does not exist")
    else:
        now = datetime.strftime(datetime.now(), '%y%m%d_%H%M%S')
        cur_save_path = os.path.join(save_path, 'results_%s' % now)
        os.mkdir(cur_save_path)

    config_filename = os.path.join(cur_save_path, 'config.txt')
    txt = open(config_filename, 'a')
    print(configs, file=txt)
    txt.close()

    txt_filename = os.path.join(cur_save_path, 'results.txt')


    #--------------- Build Train Model and Train/Test ---------------#
    set_seed(777)

    cur_train_save_path = os.path.join(cur_save_path, base_name)
    os.makedirs(cur_train_save_path, exist_ok=True)
    dataloader = data_loader(base_name, args.backbone)

    base_model = models_builder(
        train_phase='train',
        checkpoint_filename=train_checkpoint,
        **configs
    )


    engine_t = Engine_Train(
        cur_save_path=cur_train_save_path,
        train_model=base_model,
        dataloader=dataloader,
        start_epoch=train_start_epoch,
        **configs
    )

    if is_skip_train is False:
        base_train_start = time.time()
        best_epoch, finetune_checkpoint = engine_t.train()
        base_train_end = time.time()
        
        base_test_start = time.time()
        train_mae, train_rmse, train_mape = engine_t.test()
        base_test_end = time.time()
    else:
        base_test_start = time.time()
        train_mae, train_rmse, train_mape = engine_t.test()
        base_test_end = time.time()

    torch.cuda.empty_cache()

    # --------------- Build Adjust Model and Train/Test ---------------#
    set_seed(777)

    cur_adjust_save_path = os.path.join(cur_save_path, adjust_name)
    os.makedirs(cur_adjust_save_path, exist_ok=True)

    # After training, rebuild the train model by loading the best checkpoint for adjust #
    base_model = models_builder(
        train_phase='adjust',
        checkpoint_filename=finetune_checkpoint,
        **configs)

    adjust_model = adjust_builder(
        train_phase='adjust',
        backbone=args.backbone,
        checkpoint_filename=adjust_checkpoint,
        **configs
    )

    engine_g = Engine_Adjust(
        cur_save_path=cur_adjust_save_path,
        finetune_model=base_model,
        adjust_model=adjust_model,
        dataloader=dataloader,
        start_epoch=adjust_start_epoch,
        **configs
    )

    if is_skip_adjust is False:
        adjust_train_start = time.time()
        engine_g.train()
        adjust_train_end = time.time()

    adjust_test_start = time.time()
    adjust_mae, adjust_rmse, adjust_mape, fine_mae, fine_rmse, fine_mape = engine_g.test()
    adjust_test_end = time.time()


    # --------------- Print Results ---------------#
    txt = open(txt_filename, 'a')
    # train results
    # one point result
    print('--------------------------Train Results of {} on {} dataset--------------------------'.format(base_name, dataset_name))
    print('--------------------------Train Results of {} on {} dataset--------------------------'.format(base_name, dataset_name), file=txt)
    for i in range(num_of_predict):
        print('Result of {:04d} points: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(i, train_mae[i], train_rmse[i], train_mape[i]))
        print('Result of {:04d} points: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(i, train_mae[i], train_rmse[i], train_mape[i]), file=txt)

    # overall results
    print('Overall Results: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(train_mae[-1], train_rmse[-1], train_mape[-1]))
    print('Overall Results: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(train_mae[-1], train_rmse[-1], train_mape[-1]), file=txt)

    # finetune results
    # one point result
    print('\n\n--------------------------Finetune Results of {} on {} dataset--------------------------'.format(base_name, dataset_name))
    print('\n\n--------------------------Finetune Results of {} on {} dataset--------------------------'.format(base_name, dataset_name), file=txt)
    for i in range(num_of_predict):
        print(
            'Result of {:04d} points: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(i, fine_mae[i], fine_rmse[i], fine_mape[i]))
        print(
            'Result of {:04d} points: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(i, fine_mae[i], fine_rmse[i], fine_mape[i]),
            file=txt)

    # overall results
    print('Overall Results: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(fine_mae[-1], fine_rmse[-1], fine_mape[-1]))
    print('Overall Results: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(fine_mae[-1], fine_rmse[-1], fine_mape[-1]), file=txt)

    # adjust results
    # one point result
    print('\n\n--------------------------Adjust Results of {} on {} dataset--------------------------'.format(base_name, dataset_name))
    print('\n\n--------------------------Adjust Results of {} on {} dataset--------------------------'.format(base_name, dataset_name), file=txt)
    for i in range(num_of_predict):
        print('Result of {:04d} points: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(i, adjust_mae[i], adjust_rmse[i], adjust_mape[i]))
        print('Result of {:04d} points: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(i, adjust_mae[i], adjust_rmse[i], adjust_mape[i]), file=txt)

    # overall results
    print('Overall Results: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(adjust_mae[-1], adjust_rmse[-1], adjust_mape[-1]))
    print('Overall Results: MAE {:04f} | RMSE {:04f} | MAPE {:04f}'.format(adjust_mae[-1], adjust_rmse[-1], adjust_mape[-1]), file=txt)
    
    print('Base train time {:04f} | Base test time {:04f}'.format(base_train_end - base_train_start, base_test_end - base_test_start))
    print('Base train time {:04f} | Base test time {:04f}'.format(base_train_end - base_train_start, base_test_end - base_test_start), file=txt)
    
    print('Adjust train time {:04f} | Adjust test time {:04f}'.format(adjust_train_end - adjust_train_start, adjust_test_end - adjust_test_start))
    print('Adjust train time {:04f} | Adjust test time {:04f}'.format(adjust_train_end - adjust_train_start, adjust_test_end - adjust_test_start), file=txt)
    
    txt.close()
