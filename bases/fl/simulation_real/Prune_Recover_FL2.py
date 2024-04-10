import argparse
import random
# Serialize the model to get the model size of different client
import pickle
import copy
from collections import deque

from bases.optim.optimizer import SGD


from utils.functional import disp_num_params

import configs.InternetSpeed as internet_speed

import os
from copy import deepcopy
from typing import Union, Type, List

import torch
import numpy as np
from utils.save_load import mkdir_save, load

from abc import ABC, abstractmethod
from timeit import default_timer as timer

from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.nn.linear import DenseLinear, SparseLinear
from utils.functional import copy_dict
import logging
import math






def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ic','--increment',
                        help="increase",
                        action='store',
                        dest='increase',
                        type=float,
                        default=0.2,
                        required=False)

    parser.add_argument('-r', '--resume',
                        help="Resume previous prototype",
                        action='store_true',
                        dest='resume',
                        default=False,
                        required=False)

    parser.add_argument('-re', '--recover',
                        help="use recover",
                        action='store_true',
                        dest='recover',
                        default=False,
                        required=False)

    parser.add_argument('-niid',
                        help="non--iid",
                        action='store_true',
                        dest='sample',
                        default=False,
                        required=False)

    parser.add_argument('-Res',
                        help="Residual",
                        action='store_true',
                        dest='Residual',
                        default=False,
                        required=False)

    parser.add_argument('-lw',
                        help="lr warm",
                        action='store_true',
                        dest='lr_warm',
                        default=False,
                        required=False)

    parser.add_argument('-ls',
                        help="lr_scheduler",
                        action='store_true',
                        dest='lr_scheduler',
                        default=False,
                        required=False)

    parser.add_argument('-i', '--interval',
                        help="interval_round",
                        action='store',
                        dest='interval',
                        type=int,
                        default=30,
                        required=False)

    parser.add_argument('-f', '--fair',
                        help="use fair",
                        action='store',
                        dest='use_fair',
                        type=str,
                        default='no_fair',
                        required=False)

    parser.add_argument('-d', '--degree',
                        help="fair_degree",
                        action='store',
                        dest='fair_degree',
                        type=float,
                        default=0.9,
                        required=False)

    parser.add_argument('-g', '--gpu',
                        help="gpu_device",
                        action='store',
                        dest='device',
                        type=str,
                        default=0,
                        required=False)
    parser.add_argument('-m', '--merge',
                        help="merge_model",
                        action='store',
                        dest='merge',
                        type=str,
                        default='sub_fed_avg',
                        required=False)

    parser.add_argument('-wd', '--weight_decay',
                        help="small client l2 norm",
                        action='store',
                        dest='weight_decay',
                        type=float,
                        default=0.0,
                        required=False)

    parser.add_argument('-md',
                        help="min density",
                        action='store',
                        dest='min_density',
                        type=float,
                        default=0.1,
                        required=False
                        )
    parser.add_argument('-ac',
                        help="accumulate weight",
                        action='store',
                        dest='accumulate',
                        type=str,
                        default='g',
                        required=False
                        )

    parser.add_argument('-clr',
                        help="control lr",
                        action='store',
                        dest='control_lr',
                        type=float,
                        default=0,
                        required=False
                        )


    parser.add_argument('-wdn',
                        help='weight decay number',
                        dest='wdn',
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument('-ft',
                        help="ft",
                        action='store',
                        dest='ft',
                        type=str,
                        default='n',
                        required=False
                        )

    parser.add_argument('-uc',
                        help="use coeff to prune",
                        action='store',
                        dest='uc',
                        type=str,
                        default='y',
                        required=False
                        )

    parser.add_argument('-esc',
                        help="use early_stop for client model",
                        action='store_true',
                        dest='esc',
                        default=True,
                        required=False)

    parser.add_argument('-ch', '--chronous',
                        help="chronous",
                        action='store',
                        dest='chronous',
                        type=str,
                        default='syn',
                        required=False)

    parser.add_argument('-stal', '--staleness',
                        help="staleness",
                        action='store',
                        dest='stal',
                        type=str,
                        default='con',
                        required=False)

    parser.add_argument('-stal_a', '--staleness_a',
                        help="staleness_a",
                        action='store',
                        dest='stal_a',
                        type=float,
                        default='0.6',
                        required=False)

    return parser.parse_args()


class ExpConfig:  # setup the config
    def __init__(self, exp_name: str, save_dir_name: str, seed: int, batch_size: int, num_local_updates: int,
                 optimizer_class: Type, optimizer_params: dict, lr_scheduler_class: Union[Type, None],
                 lr_scheduler_params: Union[dict, None], use_adaptive: bool, device=None):
        self.exp_name = exp_name
        self.save_dir_name = save_dir_name
        self.seed = seed
        self.batch_size = batch_size
        self.num_local_updates = num_local_updates
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.lr_scheduler_class = lr_scheduler_class
        self.lr_scheduler_params = lr_scheduler_params
        self.use_adaptive = use_adaptive
        self.device = device

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, num=False, client_num=10):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        if num: self.num = 1+client_num
        else: self.num = 1

        self.patience = [patience]*self.num
        self.verbose = verbose
        self.counter = [0]*self.num

        self.best_score = [None]*self.num
        from collections import deque
        self.old_score = [deque(maxlen=(patience//2)) for i in range(self.num)]
        self.average_score = [None]*self.num
        self.early_stop = [False]*self.num
        self.val_loss_min = np.Inf
        self.delta = delta
        self.state = [None]*self.num
        self.last_score = [None]*self.num

    def __call__(self, val_loss, logger):
        '''
            功能：早停法 计算函数
            input:
                val_loss         验证损失
                model            模型
                model_path       模型保存地址
        '''
        score = val_loss
        self.state = [None]*self.num
        for i in range(self.num):

            #
            # self.old_score[i].append(score[i])
            #
            # self.average_score[i] = sum(self.old_score[i])/len(self.old_score[i])

            if self.best_score[i] is None:
                self.best_score[i] = score[i]
                self.state[i] = True
                self.last_score[i] = score[i]
                continue
            elif self.last_score[i] == score[i]:
                self.state[i] = True
                self.last_score[i] = score[i]
                logger.info(
                    f'statu {self.num}: client {i} did not receive sufficient training , current score is {score[i]}')
                continue


            elif score[i] <= self.best_score[i] + self.delta:
                self.last_score[i] = score[i]
                self.counter[i] += 1

            # if self.counter >= self.patience and score >= self.average_score:
            #     logger.info('out of the patience, but score is big than average_score')
            #     print('out of the patience, but score is big than average_score')
                if self.counter[i] >= self.patience[i]:
                    logger.info(f'statu {self.num}: client {i} out of patience, best score is {self.best_score[i]}, current score is {score[i]}')
                    print(f'statu {self.num}: client {i} out of patience, best score is {self.best_score[i]}, current score is {score[i]}')
                    self.early_stop[i] = True
                    self.state[i] = False
                    continue
                logger.info(f'statu {self.num}: client {i} EarlyStopping counter: {self.counter[i]} out of {self.patience[i]}, best score is {self.best_score[i]}, current score is {score[i]}')
                print(f'statu {self.num}: client {i} EarlyStopping counter: {self.counter[i]} out of {self.patience[i]}, best score is {self.best_score[i]}, current score is {score[i]}')

            else:
                self.last_score[i] = score[i]
                self.best_score[i] = score[i]
                self.counter[i] = 0
                self.state[i] = True
        return self.state



# experiments/FEMNIST/adaptive.py -a -i -s 0 -e
class FedMapServer(ABC):
    def __init__(self, config, args, model, seed, optimizer_class: Type, optimizer_params: dict,
                 use_adaptive, use_evaluate=True, lr_scheduler_class=None, lr_scheduler_params=None, control=None,
                 control_scheduler=None, resume=False, init_time_offset=0, device=None):
        self.config = config

        self.experiment_name = args.experiment_name
        self.recover = args.recover
        self.save_path = os.path.join("results", config.EXP_NAME, args.experiment_name)
        self.save_interval = 50
        self.use_adaptive = args.use_adaptive
        self.client_selection = args.client_selection
        self.download_speed = internet_speed.high_download_speed
        self.upload_speed = internet_speed.high_upload_speed

        self.exp_config = ExpConfig(self.config.EXP_NAME, self.save_path, seed, self.config.CLIENT_BATCH_SIZE,
                                    self.config.NUM_LOCAL_UPDATES, optimizer_class, optimizer_params,
                                    lr_scheduler_class,
                                    lr_scheduler_params, use_adaptive)


        self.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)



        self.indices = None

        self.client_is_sparse = False

        self.ip_train_loader = None
        self.ip_test_loader = None
        self.ip_optimizer_wrapper = None
        self.ip_control = None
        self.round = None

        self.test_loader = None
        self.control = None
        self.init_test_loader()
        self.init_clients()

        self.init_control()
        self.init_ip_config()
        self.save_exp_config()
        self.start_time = timer()
        self.min_density = args.min_density
        self.list_extra_params = self.get_init_extra_params()
        self.list_mask = None
        self.model_idx = None
        self.interval = args.interval
        self.merge = args.merge
        self.sever_to_client_sum = []
        self.fed_avg_acc = []
        self.model_G = []
        self.fed_avg_loss = []
        self.accumulate = args.accumulate
        self.increse = float(args.increase)
        self.use_coeff = False if args.uc == 'n' else True
        self.display = False
        self.logger = None
        self.old_list_mask = None
        self.old_accumulate_weight_dict = None
        self.list_loss = []
        self.list_acc = []
        self.fed_avg_acc = []
        self.fed_avg_loss = []
        self.list_est_time =[]
        self.list_model_size = []
        self.list_client_size =[]

        self.sever_to_client_sum =[]

        self.model_size = []
        self.list_optimizer = None
        self.list_lr_scheduler = None
        self.early_stop = False
        self.client_density = copy.deepcopy(config.client_density)
        self.list_client_density = [[] for _ in range(len(self.client_density))]

        self.list_client_loss = [[] for _ in range(len(self.client_density))]
        self.list_client_acc =  [[] for _ in range(len(self.client_density))]
        self.early_stoping = EarlyStopping(patience=self.config.patience, num=args.esc, client_num=len(self.client_density))

        self.new_mask = None
        self.lr_warm = args.lr_warm



        self.num = 0
        self.list_sparse_state_dict = None
        self.time = []
        self.train_number = None
        self.list_client_dict =[None for _ in range(len(self.client_density))]
        self.sub_density = [0 for _ in range(len(self.client_density))]
        self.list_sparse_client_dict =None
        self.list_state_dict = None
        self.first_stage = True
        self.list_stalness = [0 for _ in range(len(self.client_density))]
        self.list_coeff = [0 for _ in range(len(self.client_density))]
        self.stal = args.stal
        self.stal_a = args.stal_a
        self.list_client_time = [[] for _ in range(len(self.client_density))]
        self.list_client_sd = [None for _ in range(len(self.client_density))]
        self.client_arrive_num = [0 for _ in range(len(self.client_density))]
        self.old_model = copy.deepcopy(self.model.state_dict())
        self.average_round_time = None
        self.interval_signal = False
        # epsilon = 0.1
        # min_samples = 2
        # from sklearn.cluster import DBSCAN
        # self.dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        # self.list_cluster_loss = [[] for _ in range(len(self.client_density))]
        # self.list_cluster_acc = [[] for _ in range(len(self.client_density))]
        # self.list_cluster_density = [[] for _ in range(len(self.client_density))]
        # self.test_cluster_acc = [[] for _ in range(len(self.client_density))]
        self.client_train_time = [[0.0] for i in range(len(self.client_density))]
        self.sum_client_train_time =  [[0.0] for i in range(len(self.client_density))]
        self.server_merge_time = [0.0]
        self.sum_server_merge_time = [0.0]
        self.client_upload_time = [[0.0] for i in range(len(self.client_density))]
        self.sum_client_upload_time = [[0.0] for i in range(len(self.client_density))]
        self.client_download_time = [[0.0] for i in range(len(self.client_density))]
        self.sum_client_download_time = [[0.0] for i in range(len(self.client_density))]
        self.density_size = {}
        self.begin_save = False
        self.server_accumulate_number =[]
        self.wast_time = []



    def save_checkpoint(self):
        '''
            功能：当验证损失减少时保存模型
            input:
                val_loss         验证损失
                model            模型
                model_path       模型保存地址
        '''
        self.list_est_time.append(timer() - self.start_time)

        checkpoint = {
                      # "self.list_loss": self.list_loss,
                      # "self.list_acc": self.list_acc,
                      # 'self.list_client_time': self.list_client_time,
                      # 'self.fed_avg_loss': self.fed_avg_loss,
                      #
                      # 'self.fed_avg_acc': self.fed_avg_acc,
                      # 'self.list_est_time': self.list_est_time,
                      'self.model': self.model,
                      'self.list_client_sd': self.list_client_sd,
                      'self.control.accumulate_weight_dict': self.control.accumulate_weight_dict,
                    #   'self.begin_save':self.begin_save,
                    # 'self.model_idx': self.model_idx,

                    'self.list_mask': self.list_mask,
                    # 'self.list_model_size': self.list_model_size,
                    # 'self.list_client_density': self.list_client_density,
                    # 'self.client_density': self.client_density,
                    # 'self.list_optimizer': self.list_optimizer,
                    # 'self.list_lr_scheduler': self.list_lr_scheduler,

                      # 'self.start_time': self.start_time,
                      # 'self.list_client_acc': self.list_client_acc,
                      # 'self.list_client_loss': self.list_client_loss,
                      #
                      #
                      #
                      #
                      # 'self.num': self.num,
                      # 'self.early_stoping':self.early_stoping,
                      # 'self.list_sparse_state_dict':self.list_sparse_state_dict,
                      # 'self.time':self.time,
                      # 'self.train_number':self.train_number,
                      # 'self.list_sparse_client_dict':self.list_sparse_client_dict,
                      #
                      # 'self.first_stage':self.first_stage,
                      'self.list_stalness':self.list_stalness,
                      'self.list_coeff':self.list_coeff,
                      #
                      # 'self.old_model':self.old_model,
                      # 'self.average_round_time':self.average_round_time,
                      # 'self.list_cluster_loss':self.list_cluster_loss,
                      # 'self.list_cluster_acc ':self.list_cluster_acc,
                      # 'self.list_ ter_density':self.list_cluster_density,
                      # 'self.client_train_time':self.client_train_time,
                      # 'self.sum_client_train_time':self.sum_client_train_time,
                      # 'self.server_merge_time':self.server_merge_time,
                      # 'self.sum_server_merge_time':self.sum_server_merge_time,
                      # 'self.client_upload_time': self.client_upload_time,
                      # 'self.sum_client_upload_time':self.sum_client_upload_time,
                      # 'self.client_download_time':self.client_download_time,
                      # 'self.sum_client_download_time':self.sum_client_download_time,
                      #   'self.server_accumulate_number':self.server_accumulate_number,
                      #   'self.wast_time':self.wast_time
                      }
        checkpoint_path = os.path.join(self.save_path, 'checkpoint.pth')

        mkdir_save(checkpoint, checkpoint_path)

    def save_display_data(self):
        mkdir_save(self.list_client_acc,os.path.join(self.save_path, 'list_client_acc.pt'))
        mkdir_save(self.list_loss, os.path.join(self.save_path, 'self.list_loss.pt'))
        mkdir_save(self.list_acc, os.path.join(self.save_path, 'self.list_acc.pt'))
        mkdir_save(self.list_client_time, os.path.join(self.save_path, 'self.list_client_time.pt'))
        mkdir_save(self.fed_avg_acc, os.path.join(self.save_path, 'fed_avg_acc.pt'))
        mkdir_save(self.list_client_loss, os.path.join(self.save_path, 'list_client_loss.pt'))
        mkdir_save(self.fed_avg_loss, os.path.join(self.save_path, 'fed_avg_loss.pt'))
        mkdir_save(self.list_client_density, os.path.join(self.save_path, 'self.list_client_density'))
        mkdir_save(self.time, os.path.join(self.save_path, 'self.time'))
        mkdir_save(self.train_number, os.path.join(self.save_path, 'self.train_number'))
        mkdir_save(self.client_train_time,os.path.join(self.save_path, 'self.client_train_time'))
        mkdir_save(self.sum_client_train_time, os.path.join(self.save_path, 'self.sum_client_train_time'))
        mkdir_save(self.server_merge_time, os.path.join(self.save_path, 'self.server_merge_time'))
        mkdir_save(self.sum_server_merge_time, os.path.join(self.save_path, 'self.sum_server_merge_time'))
        mkdir_save(self.client_upload_time, os.path.join(self.save_path, 'self.client_upload_time'))
        mkdir_save(self.sum_client_upload_time, os.path.join(self.save_path, 'self.sum_client_upload_time'))
        mkdir_save(self.client_download_time, os.path.join(self.save_path, 'self.client_download_time'))
        mkdir_save(self.sum_client_download_time, os.path.join(self.save_path, 'self.sum_client_download_time'))
        mkdir_save(self.server_accumulate_number, os.path.join(self.save_path, 'self.server_accumulate_number'))






    def get_save_dir_name(self):
        if not self.use_adaptive:
            return "conventional"
        else:
            mdd_100, chl = 100 * self.config.MAX_DEC_DIFF, self.config.ADJ_HALF_LIFE
            lrhl = self.config.LR_HALF_LIFE if hasattr(self.config, "LR_HALF_LIFE") else None
            assert mdd_100 - int(mdd_100) == 0
            return "mdd{}_chl{}_lrhl{}".format(int(mdd_100), lrhl, chl)

    @abstractmethod
    def get_init_extra_params(self) -> List[tuple]:
        pass

    @abstractmethod
    def init_test_loader(self):
        pass
    def resume(self,resume,client_list):
        if resume:
            print("Resuming server...")
            checkpoint = load(os.path.join(self.save_path, 'checkpoint.pth'))
            for var_name, value in checkpoint.items():
                if var_name.startswith('self.'):
                    var_name = var_name[5:]  # 去除 'self.' 前缀
                    setattr(self, var_name, value)
            self.control.accumulate_weight_dict = checkpoint['self.control.accumulate_weight_dict']
            for key,value in self.control.accumulate_weight_dict.items():
                self.control.accumulate_weight_dict[key] = self.control.accumulate_weight_dict[key].to(self.device)
            for mask in self.list_mask:
                for key, value in mask.items():
                    mask[key] = value.to(self.device)

            len_model_size = len(self.fed_avg_acc)

            self.round = (len_model_size - 1) * self.config.EVAL_DISP_INTERVAL

            self.start_time = timer() - self.list_est_time[-1]
            self.model = self.model.to(self.device)
            self.control.model = self.model
            list_state_dict, sum_time = self.split_model(timer())
            print('current_idx: '+str(self.round))
            print("Server resumed")
            model_state_dict = []
            list_train_loader = []
            for i in range(len(client_list)):

                model_state_dict.append(list_state_dict[i])

                # for key in state_dict_key:
                #     if model_state_dict[i][key].is_sparse:
                #         model_state_dict[i][key] = model_state_dict[i][key].to_dense()
            # I think maybe need
                remaining_batches = (len(self.list_client_acc[i]) + 1) * self.exp_config.num_local_updates * self.interval
                used_sampler = client_list[i].train_loader.sampler

                num_batches_epoch = len(client_list[i].train_loader)

                used_sampler.sequence = used_sampler.sequence[remaining_batches:]

                while remaining_batches >= num_batches_epoch:
                    remaining_batches -= num_batches_epoch
                print('dataloader skip epoch')
                from torch.utils.data import DataLoader, SubsetRandomSampler
                from bases.vision.load import get_data_loader
                train_data_loader = get_data_loader(self.config.EXP_NAME,data_type="train", batch_size=client_list[0].train_loader.batch_size,
                                            sampler=used_sampler, num_workers=self.config.train_num, pin_memory=True)
                list_train_loader.append(train_data_loader)

            for i in range(len(client_list)):
                client = client_list[i]
                client.optimizer_wrapper.optimizer.load_state_dict(self.list_optimizer[i])
                if client.optimizer_wrapper.lr_scheduler is not None: client.optimizer_wrapper.lr_scheduler.load_state_dict(self.list_lr_scheduler[i])
                client.load_state_dict([self.round, model_state_dict[i]])
                client.init_train_loader(list_train_loader[i])


    @abstractmethod
    def init_clients(self):
        pass

    @abstractmethod
    def init_control(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_exp_config(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_ip_config(self, *args, **kwargs):
        pass

    def check_client_to_sparse(self):  # if model.density() <= config.TO_SPARSE_THR ,set the model to sparse
        if not self.client_is_sparse and self.model.density() <= self.config.TO_SPARSE_THR:
            self.client_is_sparse = True

    def is_one_before_adj_round(self) -> bool:
        return self.is_adj_round(self.round + 1)

    def is_adj_round(self, rd=None) -> bool:
        if rd is None:
            rd = self.round
        return self.use_adaptive and rd > 0 and rd % self.config.ADJ_INTERVAL == 0

    def get_real_size(self, list_state_dict,exp, density):
        list_model_size = []

        for i in range(len(list_state_dict)):
            ds = self.config.EXP_NAME+str(density[i])
            if ds in self.density_size.keys():
                list_model_size.append(self.density_size[ds])
            else:
                client_model = list_state_dict[i]


                if client_model == None or density[i] == 0:
                    with open((exp + '.pkl'), 'wb') as f:
                        pickle.dump(client_model, f)
                    file_size = os.path.getsize(exp + '.pkl')
                    file_size = file_size / (1024 * 1024)
                    os.remove(exp + '.pkl')
                    self.density_size['zero'] = file_size

                    list_model_size.append(0)
                    self.density_size[ds] = 0
                    continue

                with open((exp + '.pkl'), 'wb') as f:
                    pickle.dump(client_model, f)
                file_size = os.path.getsize(exp + '.pkl')
                file_size = file_size / (1024 * 1024)
                if self.config.EXP_NAME == "MNIST" :
                    file_size = (file_size)*100
                file_size = round(file_size, 4)
                list_model_size.append(file_size)
                self.density_size[ds] = file_size
                os.remove(exp + '.pkl')
        return list_model_size
    def clean_dict_to_client(self, state_dict) -> dict:
        """
        Clean up state dict before processing, e.g. remove entries, transpose.
        To be overridden by subclasses.
        """
        with torch.no_grad():
            clean_state_dict = state_dict  # not deepcopy


            for layer, prefix in zip(self.model.param_layers, self.model.param_layer_prefixes):
                key = prefix + ".bias"
                if isinstance(layer, DenseLinear) and key in clean_state_dict.keys():

                    if clean_state_dict[key].sum() != 0:
                        clean_state_dict[key] = clean_state_dict[key].view((-1, 1))
                    else:
                        clean_state_dict[key] = None


        return clean_state_dict

    @torch.no_grad()
    def process_state_dict_to_client(self, list_state_dict):
        """
        Process list_state_dict before sending to client, e.g. to cpu, to sparse, keep values only.
        if not self.client_is_sparse: send dense
        elif self.is_adj_round(): send full sparse state_dict
        else: send sparse values only
        To be overridden by subclasses.
        """
        list_state_dict = copy.deepcopy(list_state_dict)

        with torch.no_grad():
            for i in range(len(list_state_dict)):
                list_state_dict[i] = self.clean_dict_to_client(list_state_dict[i])

            for clean_state_dict in list_state_dict:
                for _, prefix in zip(self.model.prunable_layers, self.model.prunable_layer_prefixes):
                    # works for both layers
                    key_w = prefix + ".weight"
                    if key_w in clean_state_dict.keys():
                        weight = clean_state_dict[key_w]
                        sparse_weight = weight.view(weight.size(0), -1).to_sparse()
                        if sparse_weight._nnz() == 0:
                            sparse_weight = None
                        clean_state_dict[key_w] = sparse_weight

        return list_state_dict

    def process_state_dict_to_client_fast(self, list_state_dict):

        return list_state_dict

    def calculate_staleness(self,list_state_dict):
        list_stalness = [0 for _ in range(len(self.client_density))]

        for i in range(len(list_state_dict)):
            idx_state_dict = list_state_dict[i]

            if idx_state_dict == None:
                continue
            else:
                idx, _ = idx_state_dict
                time = self.round - idx + 1
                if time <= 0: time = 1.0
                if self.stal.lower().startswith('con'):
                    list_stalness[i] = 1.0
                elif self.stal.lower().startswith('poly'):
                    a = self.stal_a
                    list_stalness[i] = math.pow(time, -a)
                elif self.stal.lower().startswith('hinge'):
                    a = self.stal_a
                    if self.round-idx <= 2:
                        list_stalness[i] = 1.0
                    else:
                        list_stalness[i] = 1.0/(a*(self.round-idx-2)+1)

        for i in range(len(list_stalness)):
            if list_stalness[i] != 0:
                self.list_stalness[i] = list_stalness[i]

        sum_client = sum(self.list_stalness)
        if sum_client != 0:
            assert sum_client != 0
            for i in range(len(list_stalness)):

                self.list_coeff[i] = self.list_stalness[i]/sum_client



    @torch.no_grad()
    def Buff_mask_fed_avg(self, list_num_proc, list_sd,
                          idx, sgrd_to_upload):  # to complete the merge model ps: fedavg
        '''
        sub_fed_avg_model = sub_fed_avg(list_num_proc, list_state_dict)
        mask = sub_fed_avg_model==0
        model[mask] =  sub_fed_avg_model[mask]
        :param list_num_proc:
        :param list_state_dict:
        :param idx:
        :return:
        '''


        self_sd = self.model.state_dict()
        dict_keys = self_sd.keys()
        list_state_dict = self.list_client_sd
        sd = copy.deepcopy(self_sd)

        for key in dict_keys:
            sum_weight = torch.zeros(size=self_sd[key].size(),device=self.device)
            sum_mask = torch.zeros(size=sum_weight.size(), device=self.device)
            if key.endswith("num_batches_tracked"):
                continue
            for coeff, npc, idx_state_dict in zip(self.list_coeff, list_num_proc, list_state_dict):
                idx, state_dict = idx_state_dict
                if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense()
                sum_weight = sum_weight + coeff * npc * state_dict[key].to(self.device)
                mask = (state_dict[key] != 0).to(self.device)
                sum_mask = sum_mask + coeff * npc * mask
            divisor = torch.where(sum_mask == 0, torch.tensor([1e-10], device=self.device), sum_mask)
            sum_weight = torch.div(sum_weight, divisor)
            mask2 = sum_weight != 0
            sd[key][mask2] = sum_weight[mask2]

        self.model.load_state_dict(sd)
        if self.interval_signal:
            if self.accumulate == 'w':
                self.control.accumulate(self.old_model, idx, max(self.interval, 10))
                self.old_model = copy.deepcopy(self_sd)
            elif self.accumulate == 'wg':
                self.control.accumulate_g(sgrd_to_upload)
            elif self.accumulate == 'g':
                self.control.accumulate_g(sgrd_to_upload)
            else:
                if self.display:
                    self.logger.info('wrong accumulate')


    @torch.no_grad()
    def fedasyn(self, list_num_proc, list_state_dict,
                                       idx, sgrad_to_upload):  # Do not merge the terrible model
        # if self.display:
        #     self.logger.info('use sub_fedavg_and_fair to merge client')
        # merged_state_dict = dict()
        self_sd = self.model.state_dict()
        dict_keys = self_sd.keys()

        client_partion = 0
        for i in range(len(list_state_dict)):
            if list_state_dict[i] != None:
                client_partion += self.list_coeff[i]
        if client_partion == 0:
            return

        sd = copy.deepcopy(self_sd)
        for key in dict_keys:
            sum_weight = torch.zeros(size=self_sd[key].size(),device=self.device)
            sum_mask = torch.zeros(size=sum_weight.size(), device=self.device)
            if key.endswith("num_batches_tracked"):
                continue
            i = 0
            for num_proc, idx_state_dict in zip(list_num_proc, list_state_dict):
                if idx_state_dict == None: continue

                idx, state_dict = idx_state_dict
                if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense()
                sum_weight = sum_weight + self.list_coeff[i] * num_proc * state_dict[key].to(self.device)
                mask = (state_dict[key] != 0).to(self.device)
                sum_mask = sum_mask + self.list_coeff[i] * mask * num_proc
                i = i+1

            divisor = torch.where(sum_mask == 0, torch.tensor([1e-10], device=self.device), sum_mask)


            sum_weight = torch.div(sum_weight, divisor)
            mk = sum_weight==0
            sum_weight = sum_weight.view(sd[key].size())
            sum_weight[mk] = sd[key][mk]
            # for num_proc, state_dict in zip(list_num_proc, list_state_dict):
            #     sum_weight = sum_weight + num_proc / total_num_proc * state_dict[key].to_dense()
            if sum_weight.sum() == 0:
                continue
            sd[key] = sum_weight*client_partion + (1-client_partion)*sd[key]

        self.model.load_state_dict(sd)
        if self.interval_signal:
            if self.accumulate == 'w':
                self.control.accumulate(self.old_model, idx, max(self.interval, 10))
            elif self.accumulate == 'wg':
                self.control.accumulate_g(sgrad_to_upload)
            elif self.accumulate == 'g':
                self.control.accumulate_g(sgrad_to_upload)
            else:
                if self.display:
                    self.logger.info('wrong accumulate')


    def fed_avg_mask_model(self, list_num_proc, list_state_dict, idx):
        '''
        we want use fed_avg to accumulate model then use mask ,
        fed_avg_model = fed_avg(list_num_proc, list_sd)
        mask = fed_avg_model==0
        model[mask] =  fed_avg_model[mask]
        :param list_num_proc:
        :param list_state_dict:
        :param idx:
        :return:
        '''

        total_num_proc = sum(list_num_proc)
        if total_num_proc == 0:
            return self.model

        fed_avg_model = copy.deepcopy(self.model)
        list_state_dict = self.list_client_sd



        list_coeff = [list_num_proc[i] * self.list_coeff[i] for i in range(len(list_state_dict))]

        self_sd = self.model.state_dict()
        dict_keys = self_sd.keys()
        sd = copy.deepcopy(self_sd)
        for key in dict_keys:
            sum_weight = torch.zeros(size=self_sd[key].size(),device=self.device)
            sum_client = 0
            if key.endswith("num_batches_tracked"):
                continue
            i = 0
            for coeff, idx_state_dict in zip(list_coeff, list_state_dict):


                idx, state_dict = idx_state_dict
                if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense()
                sum_weight = sum_weight + coeff * state_dict[key].to(self.device)
                sum_client = sum_client + coeff
                i = i+1

            sum_weight = torch.div(sum_weight, sum_client)
            mk = sum_weight == 0
            sum_weight = sum_weight.view(sd[key].size())
            sum_weight[mk] = sd[key][mk]

            if sum_weight.sum() == 0:
                continue

            sd[key] = sum_weight

        fed_avg_model.load_state_dict(sd)

        return fed_avg_model


    @torch.no_grad()
    def fed_avg_model(self, list_num_proc, list_state_dict, idx):
        '''
        fed_avg_model = fed_avg(list_num_proc, list_state_dict)
        model =  fed_avg_model
        :param list_num_proc:
        :param list_state_dict:
        :param idx:
        :return:
        '''
        total_num_proc = sum(list_num_proc)
        if total_num_proc == 0:
            return self.model
        fed_avg_model = copy.deepcopy(self.model)
        list_state_dict = self.list_client_sd

        coeff = [list_num_proc[i] * self.list_coeff[i] for i in range(len(list_state_dict))]

        total_coeff = sum(coeff)

        with torch.no_grad():
            for key, param in fed_avg_model.state_dict().items():
                avg_inc_val = None
                i = 0
                for num_proc, idx_state_dict in zip(list_num_proc, list_state_dict):
                    if idx_state_dict == None:
                        continue
                    idx,state_dict = idx_state_dict
                    if key in state_dict.keys():
                        if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense()
                        state_dict[key] = state_dict[key].view(param.size())
                        mask = state_dict[key] != 0
                        if mask is None:
                            inc_val = state_dict[key].to(self.device) - param
                        else:
                            inc_val = state_dict[key].to(self.device) - param
                        inc_val.view(param.size())

                        if avg_inc_val is None:
                            avg_inc_val = coeff[i] / total_coeff * inc_val
                        else:
                            avg_inc_val += coeff[i] / total_coeff * inc_val
                    i = i+1

                if avg_inc_val is None or key.endswith("num_batches_tracked"):
                    continue
                else:
                    param.add_(avg_inc_val)

        return fed_avg_model

    @torch.no_grad()
    def fed_avg(self, list_num_proc, list_state_dict, idx, sgrd_to_upload):  # to complete the merge model ps: fedavg
        # # if self.display:
        # #     self.logger.info('use fedavg to merge client')
        # # it`s for syn fed_avg
        # coeff = [list_num_proc[i] * self.list_coeff[i] for i in range(len(list_state_dict))]
        #
        # for i in range(len(list_state_dict)):
        #     if list_state_dict[i] == None:
        #         coeff[i] = 0
        #
        # total_coeff = sum(coeff)
        # self.old_model = copy.deepcopy(self.model.state_dict())
        #
        # with torch.no_grad():
        #     for key, param in self.model.state_dict().items():
        #         avg_inc_val = None
        #         i = 0
        #         for num_proc, idx_state_dict in zip(list_num_proc, list_state_dict):
        #             if idx_state_dict == None:
        #                 continue
        #             idx,state_dict = idx_state_dict
        #             if key in state_dict.keys():
        #                 if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense()
        #                 state_dict[key] = state_dict[key].view(param.size())
        #                 mask = state_dict[key] != 0
        #                 if mask is None:
        #                     inc_val = state_dict[key].to(self.device) - param
        #                 else:
        #                     inc_val = state_dict[key].to(self.device) - param
        #                 inc_val.view(param.size())
        #
        #                 if avg_inc_val is None:
        #                     avg_inc_val = coeff[i] / total_coeff * inc_val
        #                 else:
        #                     avg_inc_val += coeff[i] / total_coeff * inc_val
        #             i = i+1
        #
        #         if avg_inc_val is None or key.endswith("num_batches_tracked"):
        #             continue
        #         else:
        #             param.add_(avg_inc_val)
        # # print(self.model.state_dict()['features.0.weight'][0])
        #
        # if self.interval_signal:
        #     if self.accumulate == 'w':
        #         self.control.accumulate(self.old_model, idx, max(self.interval, 10))
        #     elif self.accumulate == 'wg':
        #         self.control.accumulate_g(sgrd_to_upload)
        #     elif self.accumulate == 'g':
        #         self.control.accumulate_g(sgrd_to_upload)
        #     else:
        #         if self.display:
        #             self.logger.info('wrong accumulate')
        # if self.display:
        #     self.logger.info('use sub_fedavg_and_fair to merge client')
        # merged_state_dict = dict()
        self_sd = self.model.state_dict()
        dict_keys = self_sd.keys()

        client_partion = 0
        for i in range(len(list_state_dict)):
            if list_state_dict[i] != None:
                client_partion += self.list_coeff[i]
        if client_partion == 0:
            return

        sd = copy.deepcopy(self_sd)
        for key in dict_keys:
            sum_weight = torch.zeros(size=self_sd[key].size(),device=self.device)
            sum_mask = torch.zeros(size=sum_weight.size(), device=self.device)
            if key.endswith("num_batches_tracked"):
                continue
            i = 0
            for num_proc, idx_state_dict in zip(list_num_proc, list_state_dict):
                if idx_state_dict == None: continue

                idx, state_dict = idx_state_dict
                if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense()
                sum_weight = sum_weight + self.list_coeff[i] * num_proc * state_dict[key].to(self.device)
                mask = (state_dict[key] != 0).to(self.device)
                sum_mask = sum_mask + self.list_coeff[i] * mask * num_proc
                i = i+1

            divisor = torch.where(sum_mask == 0, torch.tensor([1e-10], device=self.device), sum_mask)


            sum_weight = torch.div(sum_weight, divisor)
            mk = sum_weight==0
            sum_weight = sum_weight.view(sd[key].size())
            sum_weight[mk] = sd[key][mk]
            # for num_proc, state_dict in zip(list_num_proc, list_state_dict):
            #     sum_weight = sum_weight + num_proc / total_num_proc * state_dict[key].to_dense()
            if sum_weight.sum() == 0:
                continue
            sd[key] = sum_weight*client_partion + (1-client_partion)*sd[key]

        self.model.load_state_dict(sd)
        if self.interval_signal:
            if self.accumulate == 'w':
                self.control.accumulate(self.old_model, idx, max(self.interval, 10))
            elif self.accumulate == 'wg':
                self.control.accumulate_g(sgrd_to_upload)
            elif self.accumulate == 'g':
                self.control.accumulate_g(sgrd_to_upload)
            else:
                if self.display:
                    self.logger.info('wrong accumulate')

    @torch.no_grad()
    def buff_fed_avg(self, list_num_proc, list_state_dict, idx, sgrd_to_upload):  # to complete the merge model ps: fedavg
        # if self.display:
        #     self.logger.info('use fedavg to merge client')
        # it`s for syn fed_avg
        coeff = [list_num_proc[i] * self.list_coeff[i] for i in range(len(list_state_dict))]
        list_state_dict = self.list_client_sd
        total_coeff = sum(coeff)
        self.old_model = copy.deepcopy(self.model.state_dict())
        with torch.no_grad():
            for key, param in self.model.state_dict().items():
                avg_inc_val = None
                i = 0
                for num_proc, idx_state_dict in zip(list_num_proc, list_state_dict):
                    if idx_state_dict == None:
                        continue
                    idx,state_dict = idx_state_dict
                    if key in state_dict.keys():
                        if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense()
                        state_dict[key] = state_dict[key].view(param.size())
                        mask = state_dict[key] != 0
                        if mask is None:
                            inc_val = state_dict[key].to(self.device) - param
                        else:
                            inc_val = state_dict[key].to(self.device) - param
                        inc_val.view(param.size())

                        if avg_inc_val is None:
                            avg_inc_val = coeff[i] / total_coeff * inc_val
                        else:
                            avg_inc_val += coeff[i] / total_coeff * inc_val
                    i = i+1

                if avg_inc_val is None or key.endswith("num_batches_tracked"):
                    continue
                else:
                    param.add_(avg_inc_val)

        if self.interval_signal:
            if self.accumulate == 'w':
                self.control.accumulate(self.old_model, idx, max(self.interval, 10))
            elif self.accumulate == 'wg':
                self.control.accumulate_g(sgrd_to_upload)
            elif self.accumulate == 'g':
                self.control.accumulate_g(sgrd_to_upload)
            else:
                if self.display:
                    self.logger.info('wrong accumulate')

    @torch.no_grad()
    def fed_avg_client_model(self, list_num_proc, list_state_dict):  # to complete the merge model ps: fedavg
        # if self.display:
        #     self.logger.info('use fedavg to merge client')
        client_partion = 0
        for i in range(len(list_state_dict)):
            if list_state_dict[i] != None:
                client_partion += self.list_coeff[i]

        model = copy.deepcopy(self.model)
        coeff = [list_num_proc[i] * self.list_coeff[i] / client_partion for i in range(len(list_state_dict))]


        for i in range(len(list_state_dict)):
            if list_state_dict[i] == None:
                coeff[i] = 0
        print(coeff)

        total_coeff = sum(coeff)
        with torch.no_grad():
            for key, param in model.state_dict().items():
                avg_inc_val = None
                i = 0
                for num_proc, idx_state_dict in zip(list_num_proc, list_state_dict):
                    if idx_state_dict == None:
                        continue
                    idx, state_dict = idx_state_dict
                    if key in state_dict.keys():
                        if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense()
                        state_dict[key] = state_dict[key].view(param.size())

                        inc_val = state_dict[key].to(self.device) - param

                        inc_val.view(param.size())

                        if avg_inc_val is None:
                            avg_inc_val = coeff[i] / total_coeff * inc_val
                        else:
                            avg_inc_val += coeff[i] / total_coeff * inc_val
                    i = i + 1

                if avg_inc_val is None or key.endswith("num_batches_tracked"):
                    continue
                else:
                    param.add_(avg_inc_val)

        return model

    @torch.no_grad()
    def load_client_dict(self, client_dict):  # to complete the merge model ps: fedavg
        client_dict = client_dict[1]
        model = copy.deepcopy(self.model)
        client_state_dict = copy.deepcopy(self.model.state_dict())
        with torch.no_grad():
            for key, param in client_state_dict.items():
                if key in client_dict.keys():
                    if client_dict[key].is_sparse: client_dict[key] = client_dict[key].to_dense()
                    client_state_dict[key] = client_dict[key].view(param.size())
        model.load_state_dict(client_state_dict)
        for layer in model.prunable_layers:
            mask = layer.state_dict()['weight'] != 0
            layer.mask.copy_(mask)
        return model

    @torch.no_grad()
    def split_model(self, start):
        list_state_dict = []
        sub_model_time = []
        for mask in self.list_mask:
            clean_model = copy.deepcopy(self.model)
            clean_state_dict = clean_model.state_dict()
            for layer, prefix in zip(self.model.prunable_layers, self.model.prunable_layer_prefixes):
                # works for both layers
                key_w = prefix + ".weight"
                if key_w in self.model.state_dict().keys():
                    weight = self.model.state_dict()[key_w]
                    w_mask = mask[key_w]
                    real_weight = (weight * w_mask)
                    clean_state_dict[key_w] = real_weight
                # 防止占用过多的内存
            clean_model.load_state_dict(clean_state_dict)
            # clean_model.to('cpu')
            for layer in clean_model.prunable_layers:
                mask = layer.state_dict()['weight'] != 0
                layer.mask.copy_(mask)
            # print(clean_model.density())
            clean_state_dict = clean_model.state_dict()
            list_state_dict.append(clean_state_dict)

            sub_model_time.append(timer()-start)
        return list_state_dict, sub_model_time

    def small_test_loader(self,subset_data_loader):
        if True:
            if self.config.EXP_NAME == "TinyImageNet": n = 10
            elif self.config.EXP_NAME == "CelebA": n = 2
            elif self.config.EXP_NAME == "FEMNIST": n = 2
            else:
                return subset_data_loader
            from torch.utils.data import DataLoader, SubsetRandomSampler
            # 计算要抽样的子集大小（假设是原数据集大小的 1/10）
            subset_size = len(self.test_loader.dataset) // n
            # 生成随机的子集索引
            indices = torch.randperm(len(self.test_loader.dataset))[:subset_size]
            # 使用 SubsetRandomSampler 创建新的数据加载器
            subset_sampler = SubsetRandomSampler(indices)

            subset_data_loader = DataLoader(dataset=self.test_loader.dataset,
                                            batch_size=self.test_loader.batch_size,
                                            sampler=subset_sampler, num_workers=self.test_loader.num_workers)
        return subset_data_loader




    def test_None_list_client_dict(self,list_sd):
        None_number = 0
        for i in range(len(list_sd)):
            sd = list_sd[i]
            if sd == None:
                None_number += 1
            else:
                # self.list_client_sd[i] = copy.deepcopy(sd)
                self.list_client_sd[i] = sd
                # list_sd[i] = self.list_client_sd[i]
                self.client_arrive_num[i] = self.client_arrive_num[i]

        if None_number == 10:
            return True
        else:
            return False

    def calc_model_params(self, model,display=False):
        sum_param_in_use = 0  # the sum of all used (model layers+bias)
        sum_all_param = 0
        for layer, layer_prefix in zip(model.prunable_layers, model.prunable_layer_prefixes):
            num_bias = 0 if layer.bias is None else layer.bias.nelement()
            layer_param_in_use = layer.mask.sum().int().item() + num_bias
            layer_all_param = layer.mask.nelement() + num_bias
            sum_param_in_use += layer_param_in_use
            sum_all_param += layer_all_param
            if display:
                print("\t{} remaining: {}/{} = {}".format(layer_prefix, layer_param_in_use, layer_all_param,
                                                          layer_param_in_use / layer_all_param))
        if display:
            print("\tTotal: {}/{} = {}".format(sum_param_in_use, sum_all_param, sum_param_in_use / sum_all_param))

        return sum_param_in_use / sum_all_param

    def merge_model(self,list_num_proc, list_sd, idx, sgrad_to_upload):

        if self.merge == 'buff_mask_fed_avg':
            self.Buff_mask_fed_avg(list_num_proc, list_sd, idx, sgrad_to_upload)
        elif self.merge == 'buff_fed_avg':
            self.buff_fed_avg(list_num_proc, list_sd,  idx, sgrad_to_upload)
        elif self.merge == 'fed_avg':
            self.fed_avg(list_num_proc, list_sd,  idx, sgrad_to_upload)
        elif self.merge == 'mask_fed_avg' or self.merge == 'heterofl':
            self.fedasyn(list_num_proc, list_sd, idx, sgrad_to_upload)
        elif self.merge == 'fedasyn':
            self.fedasyn(list_num_proc, list_sd, idx, sgrad_to_upload)
        elif self.merge == 'fedfix':
            self.fedasyn(list_num_proc, list_sd, idx, sgrad_to_upload)

    def test_cluster(self,i):
        standard = self.test_cluster_acc[0]

        testlength = 5
        if self.config.EXP_NAME == "TinyImageNet":
            testlength = 3
        if len(standard) < 15:
            return True

        i_acc = self.test_cluster_acc[i][-testlength:]
        standard_acc = self.test_cluster_acc[0][-testlength:]

        md = self.config.acc_sign
        if self.first_stage:
            md = self.config.acc_sign+0.01


        for idx in range(len(i_acc)):
            if standard_acc[idx]-i_acc[idx] > md:
                continue
            else:
                return True

        self.test_cluster_acc[i].append(1.0)

        return False



    def main(self, idx, list_sd, list_num_proc, lr, list_client_loss,list_client_acc, list_client_sum, round_time, list_loss, list_acc, list_est_time,
             list_model_size, is_adj_round, client_density, list_optimizer,list_lr_scheduler,train_number,interval_signal,average_round_time,sgrad_to_upload,list_time):
        self.client_train_time, self.sum_client_train_time, self.server_merge_time, self.sum_server_merge_time,\
            self.client_upload_time, self.sum_client_upload_time, self.client_download_time, self.sum_client_download_time,self.wast_time = list_time
        self.average_round_time = average_round_time
        self.interval_signal = interval_signal
        sum_time = 0
        self.train_number = train_number

        self.round = idx
        #if the first stage, client_density may be changed by FedMP_FL

        self.client_density = copy.deepcopy([round(cd, 2) for cd in client_density])

        if self.test_None_list_client_dict(list_sd) and not interval_signal:
            return self.list_state_dict, self.model_idx, sum_time, self.list_client_loss, self.sub_density, self.list_sparse_state_dict, self.list_sparse_client_dict

        self.calculate_staleness(list_sd)
        server_start = timer()
        list_sparse_state_dict = None

        #  to complete the fed avg
        self.merge_model(list_num_proc, list_sd, idx, sgrad_to_upload)


        # Evaluate the client model When the client model arrives
        subset_data_loader = self.small_test_loader(self.test_loader)
        if self.min_density < 0.99:
            test_client_interval = self.interval
        else:
            test_client_interval = self.interval*5

        sum_time += timer() - server_start

        for i in range(len(self.client_density)):

            if self.train_number[i] % test_client_interval == 1:
                if list_sd[i] == None:
                    continue
                client_model = self.load_client_dict(list_sd[i])
                client_loss, client_acc = client_model.evaluate(subset_data_loader)
                self.list_client_acc[i].append(client_acc)
                self.list_client_loss[i].append(client_loss)
                self.list_client_time[i].append(round_time)
                self.list_client_density[i].append(client_density[i])


        if interval_signal:
            loss, acc = self.model.evaluate(subset_data_loader)
            if self.min_density < 0.99:
                fed_avg_model = self.fed_avg_model(list_num_proc, list_sd, idx)

                avg_loss, avg_acc = fed_avg_model.evaluate(subset_data_loader)
            else:
                avg_loss = loss
                avg_acc = acc
            self.list_loss.append(loss)
            self.list_acc.append(acc)
            self.fed_avg_acc.append(avg_acc)
            self.fed_avg_loss.append(avg_loss)
            self.time.append(round_time)
            # for obtain the cluster acc
            # labels = self.dbscan.fit_predict(torch.tensor(client_density).reshape(-1, 1))
            # if self.min_density < 0.99:
            #     for i in range(max(labels) + 1):
            #         list_sd_i = [self.list_client_sd[i] if labels[j] == i else None for j in range(len(labels))]
            #         client_model = self.fed_avg_client_model(list_num_proc, list_sd_i)
            #         cluster_loss, cluster_acc = client_model.evaluate(subset_data_loader)
            #         density = client_model.density()
            #         for j in range(len(labels)):
            #             if labels[j] == i:
            #                 self.list_cluster_loss[j].append(cluster_loss)
            #                 self.list_cluster_acc[j].append(cluster_acc)
            #                 self.test_cluster_acc[j].append(cluster_acc)
            #                 self.list_cluster_density[j].append(density)
            # else:
            #     for j in range(len(labels)):
            #         self.list_cluster_loss[j].append(avg_loss)
            #         self.list_cluster_acc[j].append(avg_acc)
            #         self.test_cluster_acc[j].append(avg_acc)
            #         self.list_cluster_density[j].append(self.min_density)


            print("Round #{} (Experiment = {}).".format(idx, self.experiment_name))
            print("Elapsed time = {}".format(self.time[-1]))
            print("fed_avg Loss/acc={}/{}   Loss/acc={}/{}".format(avg_loss,avg_acc,loss, acc))
            print('the density is ' + ",".join([f"{client_density:.2f}" for client_density in self.client_density]))
            print('self.train_number :' + str(self.train_number))
            print('client_acc :'+",".join([f"{client_acc[-1]:.2f}" for client_acc in self.list_client_acc]))
            # print('cluster_acc :'+",".join([f"{cluster_acc[-1]:.2f}" for cluster_acc in self.list_cluster_acc]))
            print('sum_round_time :' + ",".join([f"{sum(round_time):.2f}" for round_time in self.average_round_time]))
            print('sum_down ：'+",".join([f"{client_download_time[-1]:.2f}" for client_download_time in self.sum_client_download_time]))
            print('sum_train ：'+",".join([f"{client_train_time[-1]:.2f}" for client_train_time in self.sum_client_train_time]))
            print('sum_upload ：'+",".join([f"{client_upload_time[-1]:.2f}" for client_upload_time in self.sum_client_upload_time]))
            print('sum_merge ：'+f"{self.sum_server_merge_time[-1]:.2f}")
            print('self.wast_time ：'+",".join([f"{client_upload_time:.2f}" for client_upload_time in self.wast_time]))

            self.logger.info("Round #{} (Experiment = {}).".format(idx, self.experiment_name))
            self.logger.info("Elapsed time = {}".format(self.time[-1]))
            self.logger.info("fed_avg Loss/acc={}/{}   Loss/acc={}/{}".format(avg_loss,avg_acc,loss, acc))
            self.logger.info(
                'the density is ' + ",".join([f"{client_density:.2f}" for client_density in self.client_density]))
            self.logger.info('self.train_number :' + str(self.train_number))
            self.logger.info('client_acc :'+",".join([f"{client_acc[-1]:.2f}" for client_acc in self.list_client_acc]))
            # self.logger.info('cluster_acc :'+",".join([f"{cluster_acc[-1]:.2f}" for cluster_acc in self.list_cluster_acc]))
            self.logger.info('sum_round_time :' + ",".join([f"{sum(round_time):.2f}" for round_time in self.average_round_time]))
            self.logger.info('sum_down :'+",".join([f"{client_download_time[-1]:.2f}" for client_download_time in self.sum_client_download_time]))
            self.logger.info('sum_train :'+",".join([f"{client_train_time[-1]:.2f}" for client_train_time in self.sum_client_train_time]))
            self.logger.info('sum_upload :'+",".join([f"{client_upload_time[-1]:.2f}" for client_upload_time in self.sum_client_upload_time]))
            self.logger.info('sum_merge ：'+f"{self.sum_server_merge_time[-1]:.2f}" )
            self.logger.info('self.wast_time ：'+",".join([f"{client_upload_time:.2f}" for client_upload_time in self.wast_time]))



        # to adjust model or split model

        if interval_signal:

            self.list_state_dict, self.model_idx, sub_model_time, self.list_mask, _, list_sparse_state_dict,client_density = self.control.sub_adjust_fast(
                client_density=self.client_density, use_coff=self.use_coeff,
                min_density=self.config.min_density, )

            # if self.display:
            #     self.logger.info('sub_adjust_time =  ' + str(timer() - sub_fedavg_time_start))

        else:
            start = timer()
            self.list_state_dict, sub_model_time = self.split_model(start)

        # don’t sparse
        self.list_state_dict = self.process_state_dict_to_client_fast(self.list_state_dict)







        # guarantee that the weights are not too unbalanced





        # decide increase model or not
        if interval_signal:
            if self.early_stoping.num == 1:
                # acc = [self.list_acc[-1]]
                acc = [self.fed_avg_acc[-1]]
            else:
                # acc = [self.list_acc[-1]] + [client_acc[-1] for client_acc in self.list_client_acc]
                acc = [self.fed_avg_acc[-1]] + [client_acc[-1] for client_acc in self.list_client_acc]



            if (not self.recover) or (self.min_density == 1):
                if self.early_stoping.patience[0] != self.config.patience * 8:

                    self.logger.info('Enter the end stage, increase the patience to obtain the best acc')
                    self.early_stoping.patience[0] = self.config.patience * 8


            state = self.early_stoping(acc, self.logger)

            if not self.begin_save:
                if self.early_stoping.counter[0] >= self.config.patience:
                    self.begin_save = True
                    self.early_stoping.counter[0] = 0
                    state[0] = None

            # for server model
            if state[0] == True and self.begin_save:
                self.list_optimizer = list_optimizer
                self.list_lr_scheduler = list_lr_scheduler
                self.save_checkpoint()
                self.save_display_data()

            elif state[0] == False and self.begin_save:
                self.logger.info('it is not the best model')
                checkpoint = load(os.path.join(self.save_path, 'checkpoint.pth'))
                self.model = checkpoint['self.model']
                self.control.accumulate_weight_dict = checkpoint['self.control.accumulate_weight_dict']
                self.list_client_sd = checkpoint['self.list_client_sd']
                self.list_mask = checkpoint['self.list_mask']
                self.list_stalness = checkpoint['self.list_stalness']
                self.list_coeff = checkpoint['self.list_coeff']
                for key, value in self.control.accumulate_weight_dict.items():
                    self.control.accumulate_weight_dict[key] = self.control.accumulate_weight_dict[key].to(self.device)
                self.model = self.model.to(self.device)
                self.control.model = self.model
                self.save_display_data()
                if self.recover:
                    if self.min_density == 1:
                        self.early_stop = True
                    else:
                        if self.min_density + self.increse < 1:
                            self.min_density = self.min_density + self.increse
                        else:
                            self.early_stoping.patience[0] = self.early_stoping.patience[0] * 5
                            self.min_density = 1
                        self.early_stoping.counter[0] = 0
                        self.early_stoping.early_stop[0] = False
                else:
                    self.early_stop = True

            #
            for i in range(len(self.client_density)):
                if self.client_density[i] < self.min_density:
                    self.client_density[i] = self.min_density
                    if len(self.early_stoping.counter) > 1:
                        self.early_stoping.counter[i] = 0
                        self.early_stoping.early_stop[i] = False

            if self.recover:
                # for client model
                old_client_desity = copy.deepcopy(self.client_density)

                for i in range(1, self.early_stoping.num):
                    # if not self.test_cluster(i-1):
                    #     state[i] = False
                    if state[i] == False:
                        self.logger.info(f'client{i - 1} is out of patience')
                        if self.client_density[i - 1] + self.increse < 1:
                            self.client_density[i - 1] = self.client_density[i - 1] + self.increse
                            self.early_stoping.counter[i] = 0
                            self.early_stoping.early_stop[i] = False
                        else:
                            self.client_density[i - 1] = 1
                            self.early_stoping.counter[i] = 0
                            self.early_stoping.early_stop[i] = False
                        self.logger.info(
                            f'for client{i - 1}, increase model client, client density {self.client_density[i - 1]} ' + str(
                                idx))




                min_density = min(self.client_density)

                if self.min_density + 0.01 < min_density:
                    self.min_density = min_density
                    if min_density == 1:
                        self.early_stoping.patience[0] = self.early_stoping.patience[0] * 2
                    self.early_stoping.counter[0] = 0
                    self.early_stoping.early_stop[0] = False

                # if client_density has been changed, use adjust function to adjust the client model
                if old_client_desity != self.client_density:
                    self.logger.info('rejust the model' + str(idx) + 'the new density is ' + str(
                        self.client_density))


                    # self.fix_model()
                    self.list_state_dict, self.model_idx, sub_model_time, self.list_mask, _, list_sparse_state_dict, client_density = self.control.sub_adjust_fast(
                        client_density=self.client_density, use_coff=self.use_coeff,
                        min_density=self.config.min_density, )


                    self.list_state_dict = self.process_state_dict_to_client_fast(self.list_state_dict)




        # sub_time = [sub_fedavg_time + ct + pt + st for ct, pt, st in zip(clean_time, process_dict_time, sub_model_time)]

        if interval_signal:
            # to calcualte the upload size
            sorted_clientdensity, sorted_clientdensity_indics = torch.sort(
                torch.tensor(self.client_density), descending=False)
            for i in range(len(sorted_clientdensity)):
                if i == 0:
                    self.sub_density[i] = sorted_clientdensity[i]
                else:
                    self.sub_density[i] = sorted_clientdensity[i]-sorted_clientdensity[i-1]

            if list_sparse_state_dict!= None:
                self.list_sparse_state_dict = self.get_real_size(self.process_state_dict_to_client(list_sparse_state_dict),self.experiment_name,self.sub_density)
                # self.list_sparse_state_dict = self.process_state_dict_to_client(list_sparse_state_dict)
            # to calculate the model_size
            # list_client_size = self.get_real_size(self.list_sparse_client_dict, self.server,
            #                                       self.server.client_density)
            self.list_sparse_client_dict = self.get_real_size(self.process_state_dict_to_client(self.list_state_dict),self.experiment_name,self.client_density)

        for i in range(len(self.list_state_dict)):
            self.list_state_dict[i] = [idx, self.list_state_dict[i]]



        l1_weight_sum = 0
        l2_weight_sum = 0


        for layer, prefix in zip(self.model.prunable_layers, self.model.prunable_layer_prefixes):
                # works for both layers
            key_w = prefix + ".weight"

            weight = torch.abs(self.model.state_dict()[key_w]).sum()

            l2_weight_sum += (weight**2).sum()
        if interval_signal:
            print('l2_weight_sum')
            print(l2_weight_sum)


        return self.list_state_dict, self.model_idx, sum_time, self.list_client_loss, self.sub_density, self.list_sparse_state_dict, self.list_sparse_client_dict


class FedMapClient:
    def __init__(self, model, config, use_adaptive, extra_params, exp_config, args, device):
        self.args = args
        self.config = config
        self.use_adaptive = use_adaptive
        self.model = deepcopy(model)
        self.model.train()
        self.optimizer = None
        self.optimizer_scheduler = None
        self.optimizer_wrapper = None
        self.train_loader = None
        self.client_is_sparse = False
        self.exp_config = exp_config
        self.lr_scheduler = None
        if self.exp_config.lr_scheduler_class is not None:
            self.lr_scheduler = self.exp_config.lr_scheduler_class(optimizer=self.optimizer,
                                                                   **self.exp_config.lr_scheduler_params)
        self.list_mask = [None for _ in range(len(self.model.prunable_layers))]

        self.is_sparse = False
        self.terminate = False
        self.device = device
        self.parse_init_extra_params(extra_params)
        self.test_loader = None
        self.scheduler = None
        self.model.to(self.device)
        self.num = 0
        self.use_lr_mask = False
        self.weight_decay = None
        self.client_idx = -1
        self.accumulated_sgrad = dict()




    @abstractmethod
    def init_optimizer(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_train_loader(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_test_loader(self,tl):
        pass

    @abstractmethod
    def parse_init_extra_params(self, extra_params):
        # Initialize train_loader, etc.
        pass

    def convert_to_sparse(self):
        self.model = self.model.to_sparse()
        old_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        self.optimizer = self.exp_config.optimizer_class(params=self.model.parameters(),
                                                         **self.exp_config.optimizer_params)
        if self.exp_config.lr_scheduler_class is not None:
            lr_scheduler_state_dict = deepcopy(self.lr_scheduler.state_dict())
            self.lr_scheduler = self.exp_config.lr_scheduler_class(optimizer=self.optimizer,
                                                                   **self.exp_config.lr_scheduler_params)
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
        self.optimizer.param_groups[0]["lr"] = old_lr
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, self.lr_scheduler)

        self.is_sparse = True

    @torch.no_grad()
    def load_state_dict(self, idx_state_dict):
        idx, state_dict = idx_state_dict
        self.client_idx = idx
        self.model.load_state_dict(state_dict)
        param_dict = dict(self.model.named_parameters())
        buffer_dict = dict(self.model.named_buffers())

        for key, param in {**param_dict, **buffer_dict}.items():
            if key in state_dict.keys():

                if state_dict[key].size() != param.size():
                    param.copy_(state_dict[key].view(param.size()))
                else:
                    param.copy_(state_dict[key])

        for layer in self.model.prunable_layers:
            mask = layer.state_dict()['weight'] != 0
            layer.mask.copy_(mask)

    @abstractmethod
    def parse_init_extra_params(self, extra_params):
        # Initialize train_loader, etc.
        pass


    def cleanup_state_dict_to_server(self,sparse_model) -> dict:
        """
        Clean up state dict before process, e.g. remove entries, transpose.
        To be overridden by subclasses.
        """
        clean_state_dict = sparse_model.state_dict()  # not deepcopy

        for layer, prefix in zip(sparse_model.param_layers, sparse_model.param_layer_prefixes):
            key = prefix + ".bias"
            if isinstance(layer, SparseLinear) and key in clean_state_dict.keys():
                clean_state_dict[key] = clean_state_dict[key].view(-1)

        del_list = []
        del_suffix = "placeholder"
        for key in clean_state_dict.keys():
            # clean_state_dict[key] = clean_state_dict[key].cpu()
            if key.endswith(del_suffix):
                del_list.append(key)

        for del_key in del_list:
            del clean_state_dict[del_key]

        return clean_state_dict

    @torch.no_grad()
    def process_state_dict_to_server(self,sparse_model) -> dict:
        """
        Process state dict before sending to server, e.g. keep values only, extra param in adjustment round.
        if not self.is_sparse: send dense
        elif self.adjustment_round: send sparse values + extra grad values
        else: send sparse values only
        To be overridden by subclasses.
        """
        clean_state_dict = self.cleanup_state_dict_to_server(sparse_model)


        for key, param in clean_state_dict.items():
            if param.is_sparse:
                clean_state_dict[key] = param._values()
        # but in our model, different client have different density, it will have different structure

        return clean_state_dict

    def load_mask(self, masks):
        self.list_mask = masks

    def check_client_to_sparse(self):  # if model.density() <= config.TO_SPARSE_THR ,set the model to sparse
        if not self.client_is_sparse and self.model.density() <= self.config.TO_SPARSE_THR:
            self.client_is_sparse = True

    def calc_model_params(self, display=False):
        sum_param_in_use = 0  # the sum of all used (model layers+bias)
        sum_all_param = 0
        for layer, layer_prefix in zip(self.model.prunable_layers, self.model.prunable_layer_prefixes):
            num_bias = 0 if layer.bias is None else layer.bias.nelement()
            layer_param_in_use = layer.mask.sum().int().item() + num_bias
            layer_all_param = layer.mask.nelement() + num_bias
            sum_param_in_use += layer_param_in_use
            sum_all_param += layer_all_param
            if display:
                print("\t{} remaining: {}/{} = {}".format(layer_prefix, layer_param_in_use, layer_all_param,
                                                          layer_param_in_use / layer_all_param))
        if display:
            print("\tTotal: {}/{} = {}".format(sum_param_in_use, sum_all_param, sum_param_in_use / sum_all_param))

        return sum_param_in_use / sum_all_param






    def train_model(self, train_rate,):
        num_proc_data = 0
        list_grad, loss = None,None
        if self.args.fair == 'u' :
            locale_update = int((1 - train_rate * self.args.fair_degree) * self.config.NUM_LOCAL_UPDATES)
            for i in range(locale_update):
                inputs, labels = self.train_loader.get_next_batch()
                list_grad, loss = self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))
                num_proc_data += len(inputs)

        else:
            for i in range(self.config.NUM_LOCAL_UPDATES):
                 inputs, labels = self.train_loader.get_next_batch()
                 list_grad, loss = self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))
                 num_proc_data += len(inputs)

        self.optimizer_wrapper.lr_scheduler_step(loss)
        return num_proc_data, loss, list_grad

    def main(self, idx, logger,train_rate):

        self.model.train()
        data_list = []
        real_density = self.calc_model_params()
        model_density = round(real_density, 4)
        # if real_density > 0.8: model_density = 0.1
        # if real_density < 0.2: model_density = 1.0
        # if self.args.control_lr != 0, alter the lr
        # lr_rate = self.control_lr(model_density)
        # # extract the train data set
        # for _ in range(self.config.NUM_LOCAL_UPDATES):  # the number of local update
        #     inputs, labels = self.train_loader.get_next_batch()
        #     data_list.append([inputs, labels])
        #
        # # in this function
        # old_lr = self.fine_tune(data_list)
        #
        # if lr_rate != 1:
        #     for param_group in self.optimizer_wrapper.optimizer.param_groups:
        #         param_group['lr'] = old_lr / lr_rate


        num_proc_data,loss,list_grad = self.train_model(train_rate)
        accumulated_grad = dict()
        for (key, param), g in zip(self.model.named_parameters(), list_grad):
            assert param.size() == g.size()  # only simulation
            if key in accumulated_grad.keys():
                accumulated_grad[key] += param.grad  # g
            else:
                accumulated_grad[key] = param.grad  # g
        with torch.no_grad():
            for key, grad in accumulated_grad.items():
                if key in self.accumulated_sgrad.keys():
                    self.accumulated_sgrad[key] += (grad ** 2) * num_proc_data
                else:
                    self.accumulated_sgrad[key] = (grad ** 2) * num_proc_data

        # sgrad_to_upload = copy.deepcopy(self.accumulated_sgrad)
        sgrad_to_upload = copy.copy(self.accumulated_sgrad)
        self.accumulated_sgrad = dict()

        lr = self.optimizer_wrapper.optimizer.param_groups[0]['lr']
        #
        state_dict_to_server = self.process_state_dict_to_server(copy.deepcopy(self.model).to_sparse())

        return self.model.state_dict(), num_proc_data, lr, model_density, state_dict_to_server, self.client_idx+1,sgrad_to_upload



class FedMapFL(ABC):
    def __init__(self, args, config, server, client_list):
        self.config = config
        self.use_ip = args.initial_pruning
        self.use_adaptive = args.use_adaptive
        self.tgt_d, self.max_d = args.target_density, args.max_density
        self.max_round = config.MAX_ROUND
        self.server = server
        self.client_list = client_list
        self.increase = args.increase
        self.interval = args.interval
        self.min_density = args.min_density
        self.resume = args.resume
        self.threshold = 10000
        self.pre_threshold = 0
        self.chronous = args.chronous

        self.list_loss, self.list_acc, self.list_est_time, self.list_model_size = [], [], [], []

        self.average_download_speed = config.average_download_speed
        self.average_upload_speed = config.average_upload_speed
        self.variance_download = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

        self.variance_upload = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        self.average_server_down = 50
        self.variance_server_down = 0.1
        self.average_server_up = 10
        self.variance_server_up = 0.1
        self.computerlr = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.client_density = config.client_density
        self.client_state = [True] * len(client_list)
        self.client_to_server = [0]*len(client_list)
        self.server_to_client = [0]*len(client_list)
        self.list_num = [0]*len(client_list)
        from sklearn.cluster import DBSCAN
        self.train_number = [0]*len(client_list)
        self.now_train_rate = [0.0]*len(client_list)
        self.client_start_work_time = [0] * len(client_list)
        self.list_client_sd = [None for _ in range(len(self.client_density))]
        self.last_round_time = [0]*len(client_list)
        self.per_round_time = [[] for i in range(len(self.client_density))]
        self.client_train_time = [[0.0] for i in range(len(self.client_density))]
        self.sum_client_train_time =  [[0.0] for i in range(len(self.client_density))]
        self.server_merge_time = [0.0]
        self.sum_server_merge_time = [0.0]
        self.client_upload_time = [[0.0] for i in range(len(self.client_density))]
        self.sum_client_upload_time = [[0.0] for i in range(len(self.client_density))]
        self.client_download_time = [[0.0] for i in range(len(self.client_density))]
        self.sum_client_download_time = [[0.0] for i in range(len(self.client_density))]
        self.waste_time = [0] * len(self.client_list)
        self.list_time = [self.client_train_time,self.sum_client_train_time,self.server_merge_time,self.sum_server_merge_time,
                          self.client_upload_time,self.sum_client_upload_time,self.client_download_time,self.sum_client_download_time,
                          self.waste_time
                          ]

        self.Residual = args.Res

        import numpy as np

        # 创建DBSCAN模型
        epsilon = 0.15
        min_samples = 1
        self.dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        self.max_size =None
        self.interval_signal = False
        self.last_idx = -1
        self.se_up = 0


        ds_path  = os.path.join("results", 'density_size.pt')
        if os.path.exists(ds_path):
            self.density_size = load(ds_path)
        else:
            self.density_size={}
            mkdir_save(self.density_size, ds_path)



    def get_real_size(self, list_state_dict,exp, density):
        list_model_size = []

        for i in range(len(list_state_dict)):
            ds = self.config.EXP_NAME+str(density[i])
            if ds in self.density_size.keys():
                list_model_size.append(self.density_size[ds])
            else:
                client_model = list_state_dict[i]


                if client_model == None or density[i] == 0:
                    with open((exp + '.pkl'), 'wb') as f:
                        pickle.dump(client_model, f)
                    file_size = os.path.getsize(exp + '.pkl')
                    file_size = file_size / (1024 * 1024)
                    os.remove(exp + '.pkl')
                    self.density_size['zero'] = file_size

                    list_model_size.append(0)
                    self.density_size[ds] = 0
                    continue

                with open((exp + '.pkl'), 'wb') as f:
                    pickle.dump(client_model, f)
                file_size = os.path.getsize(exp + '.pkl')
                file_size = file_size / (1024 * 1024)
                if self.config.EXP_NAME == "MNIST" :
                    file_size = (file_size)*100
                file_size = round(file_size,4)
                # else:
                #     file_size = file_size-self.density_size['zero']
                list_model_size.append(file_size)
                self.density_size[ds] = file_size
                os.remove(exp + '.pkl')
        return list_model_size

    def get_global_min_max(self,state_dict):


        all_values = torch.cat([value.flatten() for value in state_dict.values()])
        global_min = torch.min(all_values)
        global_max = torch.max(all_values)
        return global_min, global_max

    def normalize_state_dict(self, state_dict, global_min, global_max):
        normalized_dict = {}
        for key, value in state_dict.items():
            normalized_value = (value - global_min) / (global_max - global_min)
            normalized_dict[key] = normalized_value
        return normalized_dict
    def accumulate_dict(self,accumulate_dict,sgrd,sum_mask):
        global_min, global_max = self.get_global_min_max(sgrd)
        normalized = self.normalize_state_dict(sgrd, global_min, global_max)
        for key in sgrd:
            accumulate_dict[key] += normalized[key]
            sum_mask[key] += normalized[key]!=0

    def zeroed_state_dict(self,state_dict):
        zeroed_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                zeroed_dict[key] = torch.zeros_like(value)
            else:
                zeroed_dict[key] = self.zeroed_state_dict(value)
        return zeroed_dict
    def get_internet_speed(self):

        download_speed, upload_speed = [], []
        import numpy as np

        server_up = 0
        server_down = 0
        n = 20
        for i in range(n):
            server_up += np.random.lognormal(mean=0, sigma=self.variance_server_up) * self.average_server_up
            server_down += np.random.lognormal(mean=0, sigma=self.variance_server_down) * self.average_server_down
        server_up = server_up/n
        server_down = server_down/n

        for i in range(len(self.average_upload_speed)):
            dp = 0
            up = 0
            for j in range(n):
                dp += np.random.lognormal(mean=0, sigma=self.variance_upload[i]) * self.average_download_speed[i]
                up += np.random.lognormal(mean=0, sigma=self.variance_download[i]) * self.average_upload_speed[i]

            dp = dp/n
            up = up/n

            download_speed.append(dp)
            upload_speed.append(up)

        if self.config.EXP_NAME == "CelebA":
            server_up = server_up*100
            server_down = server_down*100

        return server_up, server_down, download_speed, upload_speed


    def main(self):
        # model initialization completed

        log_file_path = os.path.join(self.server.save_path, 'app.log')
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(message)s',
                            handlers=[
                                logging.FileHandler(log_file_path),  # 写入文件
                                # logging.StreamHandler()  # 输出到控制台
                            ])
        # 创建一个日志记录器
        logger = logging.getLogger(__name__)

        idx = 0
        self.server.logger = logger
        self.communicate_time_from_server_to_client = [0] * len(self.client_list)
        start = timer()
        self.server.start_time = start
        self.round_time = 0
        self.list_sparse_client_dict = None
        self.last_train_num = [0] * len(self.client_list)
        self.average_round_time = [deque(maxlen=self.interval) for _ in range(len(self.client_list))]

        start_client_idx = 10
        if self.resume:
            self.server.resume(self.resume, self.client_list)
            idx = self.server.round+1
            self.client_density = self.server.list_client_density[-1]
            self.round_time = self.server.time[-1]
            self.list_sparse_client_dict = self.server.list_sparse_client_dict
            self.train_number = self.server.train_number

            self.communicate_time_from_server_to_client = [self.server.time[-1]] * len(self.client_list)

            self.average_round_time = self.server.average_round_time

        self.round_time = 0
        list_client_loss, list_client_acc, list_client_sum, list_model_size =  [], [], [], []

        # the time from server to client



        list_store_model_server = [None for _ in range(len(self.client_density))]
        list_store_model_client = [None for _ in range(len(self.client_density))]
        self.one_round_time = 0
        idx = self.train_number[0]

        accum_sgrad = self.zeroed_state_dict(self.server.model.state_dict())
        sum_mask = copy.deepcopy(accum_sgrad)
        model = copy.deepcopy(self.server.model)
        model.load_state_dict(accum_sgrad)
        dict = self.server.process_state_dict_to_client([model.state_dict()])
        self.get_real_size([dict], self.server.experiment_name,[0])
        begin_time_client_upload = [0]*len(self.client_state)
        begin_time_server_merge = 0

        begin_time_client_download = [0]*len(self.client_state)
        debug = False
        while True:

            if idx % self.interval == 0 and idx != self.last_idx:
                self.interval_signal = True
            else:
                self.interval_signal = False


            self.last_idx = idx

            if self.server.early_stop:
                logger.info('early stop at'+str(idx) + ' round of FL')
                mkdir_save(self.server.list_client_acc, os.path.join(self.server.save_path, 'list_client_acc.pt'))
                mkdir_save(self.server.fed_avg_acc, os.path.join(self.server.save_path, 'fed_avg_acc.pt'))
                mkdir_save(self.server.list_client_loss, os.path.join(self.server.save_path, 'list_client_loss.pt'))
                mkdir_save(self.server.fed_avg_loss, os.path.join(self.server.save_path, 'fed_avg_loss.pt'))
                mkdir_save(self.server.model_G, os.path.join(self.server.save_path, 'model_G.pt'))
                break



            list_state_dict, list_accum_sgrad, list_last_lr, sparse_m= [], [], [], []

            is_adj_round = False




            density = []

            for i in range(len(self.client_list)):

                if self.client_state[i] == False:
                    assert self.server_to_client[i] > 0 or self.client_to_server[i] > 0
                    assert self.server_to_client[i] == 0 or self.client_to_server[i] == 0
                    #it means the old model is not sent to the server or the client do not get the new model.
                    #be + excessively indictes the passive tense.

                else:
                    self.client_state[i] = True

            for i in range(len(self.client_list)):
                if self.client_state[i]:
                    client_time = timer()
                    client = self.client_list[i]
                    sd, npc, last_lr, ds, sparse_model,client_idx, sgrd  = client.main(idx, logger, self.now_train_rate[i])
                    self.client_train_time[i].append((timer() - client_time) * self.computerlr[i])
                    self.sum_client_train_time[i].append(self.sum_client_train_time[i][-1]+self.client_train_time[i][-1])
                    list_store_model_server[i] = [client_idx, sd]
                    list_state_dict.append([client_idx, sd])
                    self.train_number[i] += 1
                    self.list_num[i] = npc # Amount of data for client-side training models

                else:
                    sd, npc, last_lr, ds, sparse_model, client_time, sgrd = None, 0, None, None, None, 0, None

                    list_state_dict.append(None)

                #when the old client model arrives at the server, it need npc to calculate the coeff,

                list_last_lr.append(last_lr)
                density.append(ds)
                sparse_m.append(sparse_model)



                if sgrd != None and len(sgrd) != 0:
                    self.accumulate_dict(accum_sgrad, sgrd, sum_mask)



            if self.interval_signal:
                for key in accum_sgrad.keys():
                    # print(sum_mask)
                    if sum_mask[key].type() == 'torch.cuda.LongTensor':
                        if sum_mask[key] == 0:
                            accum_sgrad[key] = 0
                        else:
                            accum_sgrad[key] = accum_sgrad[key]/sum_mask[key]
                        continue
                    divisor = torch.where(sum_mask[key] == 0, torch.tensor([1e-10], device=sum_mask[key].device), sum_mask[key])
                    accum_sgrad[key] = torch.div(accum_sgrad[key], divisor)
                # self.sgrad_to_upload = copy.deepcopy(accum_sgrad)
                self.sgrad_to_upload = accum_sgrad
                # print(f'first{self.sgrad_to_upload}')
                accum_sgrad = self.zeroed_state_dict(self.server.model.state_dict())
                sum_mask = copy.deepcopy(accum_sgrad)

            else:
                self.sgrad_to_upload = None


            last_lr = list_last_lr[0]

            model_size = self.get_real_size(sparse_m, self.server.experiment_name, density)
            # print(f'model_size   {model_size}')
            # print(density)
            if self.max_size==None:
                self.max_size = self.density_size[self.config.EXP_NAME + str(1.0)]
            # if display:
            #     logger.info('client_to_server_model_size  :'+str(model_size))
            # print(model_size)
            self.server.list_model_size.append(model_size)
            for i in range(len(self.client_list)):
                if not self.client_state[i]:
                    assert self.server_to_client[i] > 0 or self.client_to_server[i] > 0
                    assert self.server_to_client[i] == 0 or self.client_to_server[i] == 0
            server_up, server_down, download_speed, upload_speed = self.get_internet_speed()



            self.client_to_server = [cs + ms for cs, ms in zip(self.client_to_server, model_size)]
            #
            if debug:
                print(' Round '+ str(idx)+'self.threshold' + str(self.threshold) + ' client_state ' + str(self.client_state))
                print('First self.client_to_server' + str(self.client_to_server))





            from control.sub_algorithm import simulate_client_to_server, determine_density, simluate_server_to_client
            time = []
            for i in range(len(self.client_state)):
                assert self.server_to_client[i] > 0 or self.client_to_server[i] > 0
                assert self.server_to_client[i] == 0 or self.client_to_server[i] == 0
                if self.client_state[i]:
                    time.append(self.communicate_time_from_server_to_client[i]+self.client_train_time[i][-1])
                    begin_time_client_upload[i] = time[i]
                else:
                    time.append(self.communicate_time_from_server_to_client[i])





            server_receive_time = np.array(simulate_client_to_server(time, copy.copy(self.client_to_server), upload_speed, server_down))

            server_close_time = copy.copy(server_receive_time)

            list_round_time = np.array([0]*len(self.client_list),dtype=float)
            for i in range(len(server_receive_time)):
                list_round_time[i] = server_receive_time[i]-self.round_time

            if self.server.merge == 'fedasyn':
                max_true_client = 1
                if start_client_idx == len(server_close_time):
                    server_dowload_sequence = list(range(0,len(server_close_time)))
                else:
                    server_dowload_sequence =  list(range(start_client_idx,len(server_close_time)))+list(range(0,start_client_idx))


            else:
                max_true_client = len(server_close_time)
                server_dowload_sequence = list(range(0, len(server_close_time)))




            for i in server_dowload_sequence:
                if self.client_state[i]:
                    if server_close_time[i] > self.threshold:
                        self.client_state[i] = False
                        server_close_time[i] = self.threshold
                        self.client_to_server[i] = self.client_to_server[i] - upload_speed[i]*(self.threshold-time[i])
                        if self.client_to_server[i] <= 0:
                            self.client_to_server[i] = 0.001
                        assert self.server_to_client[i] > 0 or self.client_to_server[i] > 0
                        assert self.server_to_client[i] == 0 or self.client_to_server[i] == 0
                    else:
                        assert server_close_time[i] <= self.threshold
                        if max_true_client > 0:
                            self.client_to_server[i] = 0
                            self.client_upload_time[i].append(server_close_time[i]-begin_time_client_upload[i])
                            self.sum_client_upload_time[i] += self.client_upload_time[i][-1]
                            max_true_client = max_true_client-1
                            start_client_idx = i+1
                        else:
                            self.client_to_server[i] = self.client_to_server[i] - upload_speed[i] * (
                                        self.threshold - time[i])
                            if self.client_to_server[i] <= 0:
                                self.client_to_server[i] = 0.001
                            self.client_state[i] = False


                else:
                    # print(self.server_to_client)
                    # print(self.client_to_server)
                    assert self.server_to_client[i] > 0 or self.client_to_server[i] > 0
                    assert self.server_to_client[i] == 0 or self.client_to_server[i] == 0
                    if self.client_to_server[i] > 0:
                        if server_close_time[i] > self.threshold:
                            self.client_to_server[i] = self.client_to_server[i] - upload_speed[i] *(self.threshold-time[i])
                            if self.client_to_server[i] < 0: self.client_to_server[i] = 0.001
                            server_close_time[i] = self.threshold
                        else:
                            assert self.server_to_client[i] == 0
                            assert list_state_dict[i] == None
                            if max_true_client > 0:
                                self.client_to_server[i] = 0
                                list_state_dict[i] = list_store_model_server[i]
                                self.client_state[i] = True
                                self.client_upload_time[i].append(server_close_time[i] - begin_time_client_upload[i])
                                self.sum_client_upload_time[i] += self.client_upload_time[i][-1]
                                max_true_client = max_true_client - 1
                                start_client_idx = i+1
                            else:
                                self.client_to_server[i] = 0.001
                                self.client_state[i] = False
                                server_close_time[i] = self.threshold
                    if self.server_to_client[i] > 0:
                        server_receive_time[i] = self.threshold+1000




            if idx < 2:
                self.round_time = max(server_close_time)
            else:

                if self.chronous.lower().startswith('syn'):
                    for i in range(len(self.last_train_num)):
                        self.average_round_time[i].append(server_close_time[i] - self.round_time)
                    self.round_time = max(server_close_time)

                else:
                    true_sum = 0
                    for i in range(len(self.last_train_num)):
                        if self.client_state[i]:
                            true_sum += 1



                    if true_sum == 1 and self.server.merge != 'fedasyn':
                        labels = self.dbscan.fit_predict(server_receive_time.reshape(-1, 1))

                        new_threshold1 = max(server_receive_time[labels == labels[
                            np.argmin(server_receive_time)]]) + self.config.asyn_interval

                        new_threshold2 = sorted(server_receive_time, reverse=False)[1]+self.config.asyn_interval
                        new_threshold = max(new_threshold1,new_threshold2)
                        new_threshold1 = min(new_threshold1, self.threshold + 2)

                        if debug:
                            print(f'new1 {new_threshold1}, new2    {new_threshold2}')
                        sum_time = new_threshold - self.threshold
                        for i in range(len(self.client_list)):
                            if not self.client_state[i]:
                                assert self.server_to_client[i] > 0 or self.client_to_server[i] > 0
                                assert self.server_to_client[i] == 0 or self.client_to_server[i] == 0
                                if self.server_to_client[i] > 0:
                                    self.server_to_client[i] = 0.001 if self.server_to_client[i] <= download_speed[
                                        i] * sum_time else \
                                        self.server_to_client[i] - download_speed[i] * sum_time
                                    server_close_time[i] = new_threshold

                                if self.client_to_server[i] > 0:
                                    if server_receive_time[i] < new_threshold:
                                        assert self.client_to_server[i] <= upload_speed[i] * sum_time
                                        server_close_time[i] = server_receive_time[i]
                                        self.client_to_server[i] = 0
                                        list_state_dict[i] = list_store_model_server[i]
                                        self.client_state[i] = True
                                        self.client_upload_time[i].append(
                                            server_close_time[i] - begin_time_client_upload[i])
                                        self.sum_client_upload_time[i] += self.client_upload_time[i][-1]
                                    else:
                                        self.client_to_server[i] -= upload_speed[i] * sum_time
                                        server_close_time[i] = new_threshold
                        self.threshold = new_threshold
                        # print(self.client_to_server)
                        # print(self.server_to_client)
                        # print(server_receive_time)
                        # print(self.client_state)
                        # print(f'self.threshold : {self.threshold}')
                        # assert sum(self.client_state) != 1
                    for i in range(len(self.last_train_num)):
                        if self.client_state[i]:
                            self.average_round_time[i].append(server_close_time[i] - self.last_round_time[i])
                            self.last_round_time[i] = server_close_time[i]

                    self.round_time = self.threshold

                    for i in range(len(self.last_train_num)):
                        if self.client_state[i]:
                            self.waste_time[i] += (self.threshold - server_close_time[i])


            if debug:
                print(f'threshold: {self.threshold},  server_close_time: {server_close_time}')
                print(f'waste_time :{self.waste_time}')
            # print(f'threshold: {self.threshold},  server_close_time: {server_close_time}')
            # print(f'waste_time :{self.waste_time}')
            # print(sum(self.client_state))
            list_optimizer = []
            list_lr_scheduler =[]
            if self.interval_signal:
                for client in self.client_list:
                    list_optimizer.append(client.optimizer_wrapper.optimizer.state_dict())
                    if client.optimizer_wrapper.lr_scheduler is not None:
                        list_lr_scheduler.append(client.optimizer_wrapper.lr_scheduler.state_dict())


                if self.server.first_stage and self.server.min_density != 1.0 and idx != 0 and self.server.merge != 'heterofl' and self.interval_signal:

                    standard = sum(self.average_round_time[0])/len(self.average_round_time[0])
                    for i in range(len(self.average_round_time)):
                        art = sum(self.average_round_time[i])/len(self.average_round_time[i])
                        rate = (art-standard)/art
                        if abs(rate) > 0.05 :
                            old_density = self.client_density[i]
                            new_td = round(self.client_density[i]*(1-0.9*rate), 2)
                            if rate > 0.05 and new_td >= self.config.min_density or new_td >= self.min_density:

                                self.client_density[i] = new_td

                                print(f'rate1: {rate}')
                                logger.info(f'rate1: {rate}')
                                logger.info(
                                    f'for client {i} Tune the client density {old_density} to {self.client_density[i]} keep the client balance  ')
                                print(
                                    f'for client {i} Tune the client density {old_density} to {self.client_density[i]} keep the client balance  ')






            if self.interval_signal:
                sgrad_to_upload = self.sgrad_to_upload
                now_train_number = [self.train_number[i] - self.last_train_num[i] for i in
                                    range(len(self.now_train_rate))]
                max_now_train_number = max(now_train_number)
                for i in range(len(self.now_train_rate)):
                    if now_train_number[i] == 0:
                        self.now_train_rate[i] = float(now_train_number[i] - max_now_train_number)
                    else:
                        self.now_train_rate[i] = float(now_train_number[i] - max_now_train_number) / now_train_number[i]
                self.last_train_num = copy.copy(self.train_number)
                print(now_train_number)
                print(self.now_train_rate)
            else:
                sgrad_to_upload = None

            if self.client_state[0]:
                begin_time_server_merge = self.round_time

            list_state_dict, model_idx, sum_time, self.client_loss, sub_density, list_sparse_state_dict,self.list_sparse_client_dict\
                = self.server.main(idx, list_state_dict, self.list_num, last_lr, list_client_loss, list_client_acc,list_client_sum,
                                   self.round_time,
                                   self.list_loss, self.list_acc, self.list_est_time,
                                   self.list_model_size,
                                   is_adj_round, self.client_density, list_optimizer, list_lr_scheduler,
                                   self.train_number,self.interval_signal, self.average_round_time,
                                   sgrad_to_upload,self.list_time)


            if self.server.first_stage and self.min_density != 1.0:
                for i in range(len(self.client_density)):
                    if not self.client_density[i] == self.server.client_density[i]:
                        self.server.first_stage = False
                        break


            self.client_density = self.server.client_density



            # start time of every round
            for i in self.client_state:
                if i:
                    self.round_time = sum_time + self.round_time
                    break

            if self.client_state[0]:
                self.server_merge_time.append(self.round_time-begin_time_server_merge)
                self.sum_server_merge_time.append(self.sum_server_merge_time[-1]+self.server_merge_time[-1])

            for i in range(len(self.client_state)):
                if self.client_state[i]:
                    begin_time_client_download[i] = self.round_time

            False_number = 0

            for st in self.client_state:
                if not st:
                    False_number += 1
            if debug:
                print('First server_to_client' + str(self.server_to_client))

            if not False_number == len(self.client_state):
                for i in range(len(self.client_list)):
                    if not self.client_state[i]:
                        assert self.server_to_client[i] > 0 or self.client_to_server[i] > 0
                        assert self.server_to_client[i] == 0 or self.client_to_server[i] == 0
                        if self.server_to_client[i] > 0:
                            self.server_to_client[i] = 0.001 if self.server_to_client[i] <= download_speed[
                                i] * sum_time else \
                                self.server_to_client[i] - download_speed[i] * sum_time

                        if self.client_to_server[i] > 0:
                            self.client_to_server[i] = 0.001 if self.client_to_server[i] <= upload_speed[
                                i] * sum_time else \
                                self.client_to_server[i] - upload_speed[i] * sum_time
                # send model to the client
                if debug:
                    print('Third self.client_to_server' + str(self.client_to_server))

            # list_upload_size = self.get_real_size(list_sparse_state_dict, self.server.experiment_name, sub_density)
            # list_client_size = self.get_real_size(self.list_sparse_client_dict, self.server.experiment_name,
            #                                       self.server.client_density)
            list_client_size = self.list_sparse_client_dict
            list_upload_size = [0]*len(list_sparse_state_dict)
            if self.Residual:
                list_upload_size = list_sparse_state_dict
            else:
                wait_load_model = []
                for i in range(len(model_idx)):
                    idx = model_idx[i][-1]
                    if self.list_sparse_client_dict[i] not in wait_load_model and self.client_state[i]:
                        list_upload_size[i] = self.list_sparse_client_dict[i]
                        wait_load_model.append(self.list_sparse_client_dict[i])

                list_upload_size = sorted(list_upload_size, reverse=False)


                new_model_idx = [0]*len(list_sparse_state_dict)
                for i in range(len(model_idx)):
                    np_list_upload_size = np.array(list_upload_size)
                    j = np.where(np_list_upload_size==self.list_sparse_client_dict[i])
                    # print(self.list_sparse_client_dict)
                    # print(self.server.client_density)
                    # print(sub_density)
                    # print(j)
                    # print(np_list_upload_size)
                    # print(self.list_sparse_client_dict[i])
                    assert len(j[0]) == 1 or len(j[0]) == 0
                    if len(j[0]) == 1:
                        temp = list(range(int(j[0]+1)))
                        new_model_idx[i] = temp
                    else:
                        assert list_upload_size[0] == 0
                        new_model_idx[i] = [0]
                model_idx = new_model_idx

                # print(model_idx)





            if self.chronous.lower().startswith('syn'):
                self.threshold = 1000 + self.round_time

            else:
                need_client = [0]
                for i in range(len(self.client_state)):
                    if self.client_state[i]:
                        need_client.append(list_client_size[i])
                if self.Residual:
                    need_upload = max(need_client)
                else:
                    need_upload = sum(list(set(need_client)))

                # print(list(set(need_client)))
                self.threshold = max(self.round_time + self.config.asyn_interval,
                                     self.round_time + (need_upload / server_up) + 0.1)

                if self.server.merge == 'fedasyn':
                    self.threshold = self.round_time + (need_upload / server_up) + 0.1


                if debug:
                    print(f'self.threshold_increment {self.threshold-self.round_time}')







            server_upload = [0.0] * len(self.client_list)

            for i in range(len(self.client_list)):
                server_upload[i] += list_upload_size[i]
                self.client_start_work_time[i] = self.server_to_client[i] / download_speed[i]
                if self.client_state[i]:
                    assert self.server_to_client[i] == 0 and self.client_to_server[i] == 0
                    self.server_to_client[i] += list_client_size[i]

            # print(f'server_upload {server_upload}')
            # print(f'client start work time {self.client_start_work_time}')
            time_from_server_to_client = simluate_server_to_client([0.0001 for i in range(len(self.client_list))],
                                                                   self.client_start_work_time, copy.copy(server_upload),
                                                                   server_up,
                                                                   model_idx, download_speed)

            # print((' time_from_server_to_client  ' + str(time_from_server_to_client)))
            for i in range(len(self.client_list)):
                if not self.client_state[i]:
                    time_from_server_to_client[i] = self.server_to_client[i] / download_speed[i]

            # print((' time_from_server_to_client  ' + str(time_from_server_to_client)))
            if debug:
                print('Second server_to_client' + str(self.server_to_client))
            for i in range(len(self.client_list)):
                if self.client_state[i]:
                    #如果客户端一直是正常的，但是这一次无法正常到达，就把模型保存起来
                    if time_from_server_to_client[i]+self.round_time > self.threshold:
                        # self.server_to_client[i] -= download_speed[i] * self.pre_threshold
                        list_store_model_client[i] = list_state_dict[i]
                        self.server_to_client[i] -= min(download_speed[i] * (self.threshold-self.round_time),server_up * (self.threshold-self.round_time))
                        if self.server_to_client[i] < 0:
                            self.server_to_client[i] = 0.001
                        self.client_state[i] = False
                        time_from_server_to_client[i] = self.threshold-self.round_time
                        assert self.server_to_client[i] > 0 or self.client_to_server[i] > 0
                        assert self.server_to_client[i] == 0 or self.client_to_server[i] == 0

                    else:
                        self.server_to_client[i] = 0
                        self.client_download_time[i].append(time_from_server_to_client[i]+self.round_time-begin_time_client_download[i])
                        self.sum_client_download_time[i] += self.client_download_time[i][-1]

                else:
                    assert self.server_to_client[i] > 0 or self.client_to_server[i] > 0
                    assert self.server_to_client[i] == 0 or self.client_to_server[i] == 0
                    if self.server_to_client[i] > 0:
                        assert self.server_to_client[i] > 0
                        if time_from_server_to_client[i] + self.round_time > self.threshold:

                            self.server_to_client[i] -= download_speed[i] * (self.threshold-self.round_time)

                            if self.server_to_client[i] < 0:
                                self.server_to_client[i] = 0.001
                            time_from_server_to_client[i] = self.threshold-self.round_time
                        else:
                            self.server_to_client[i] = 0

                            time_from_server_to_client[i] = self.client_start_work_time[i]
                            assert self.server_to_client[i] == 0 and self.client_to_server[i] == 0
                            self.client_state[i] = True
                            self.client_download_time[i].append(
                                time_from_server_to_client[i] + self.round_time - begin_time_client_download[i])
                            self.sum_client_download_time[i] += self.client_download_time[i][-1]
                            list_state_dict[i] = list_store_model_client[i]

                    if self.client_to_server[i] > 0:
                        assert self.server_to_client[i] == 0
                        time_from_server_to_client[i] = 0
            if debug:
                print('Third server_to_client' + str(self.server_to_client))

            for i in range(len(self.client_state)):
                if self.client_state[i]:
                    client = self.client_list[i]
                    client.load_state_dict(list_state_dict[i])
                    if self.interval_signal:
                        client.optimizer_wrapper.lr_scheduler_step(self.server.list_client_acc[i][-1])
            if debug:
                print((' time_from_server_to_client  ' + str(time_from_server_to_client)))
            self.communicate_time_from_server_to_client = [self.round_time + sc for sc in time_from_server_to_client]
            idx = self.train_number[0]











