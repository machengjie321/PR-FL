import argparse
import random
# Serialize the model to get the model size of different client
import pickle
import copy
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






def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ic',
                        help="increase",
                        action='store',
                        dest='increase',
                        type=str,
                        required=True)

    parser.add_argument('-r', '--resume',
                        help="Resume previous prototype",
                        action='store_true',
                        dest='resume',
                        default=False,
                        required=False)


    parser.add_argument('-i', '--interval',
                        help="interval_round",
                        action='store',
                        dest='interval',
                        type=int,
                        required=True)

    parser.add_argument('-f', '--fair',
                        help="use fair",
                        action='store',
                        dest='use_fair',
                        type=str,
                        required=True)

    parser.add_argument('-d', '--degree',
                        help="fair_degree",
                        action='store',
                        dest='fair_degree',
                        type=float,
                        required=True)

    parser.add_argument('-g', '--gpu',
                        help="gpu_device",
                        action='store',
                        dest='device',
                        type=str,
                        required=True)
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
                        required=True)

    parser.add_argument('-md',
                        help="min density",
                        action='store',
                        dest='min_density',
                        type=float,
                        required=True
                        )
    parser.add_argument('-ac',
                        help="accumulate weight",
                        action='store',
                        dest='accumulate',
                        type=str,
                        required=True
                        )
    parser.add_argument('-clr',
                        help='lr decay or lr increase',
                        action='store',
                        dest='control_lr',
                        type=float,
                        required=True)



    parser.add_argument('-wdn',
                        help='weight decay number',
                        dest='wdn',
                        type=int,
                        required=True)


    parser.add_argument('-ft',
                        help="ft",
                        action='store',
                        dest='ft',
                        type=str,
                        required=True
                        )

    parser.add_argument('-uc',
                        help="use coeff to prune",
                        action='store',
                        dest='uc',
                        type=str,
                        default='y',
                        required=False
                        )







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
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        from collections import deque
        self.old_score = deque(maxlen=(patience//2))
        self.average_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss,logger):
        '''
            功能：早停法 计算函数
            input:
                val_loss         验证损失
                model            模型
                model_path       模型保存地址
        '''
        score = val_loss
        self.old_score.append(score)
        self.average_score = sum(self.old_score)/len(self.old_score)
        if self.best_score is None:
            self.best_score = score
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience and score > self.average_score:
                logger.info('out of the patience, but score is big than average_score')
                print('out of the patience, but score is big than average_score')
            elif self.counter >= self.patience and score < self.average_score:
                logger.info('out of patience')
                print('out of patience')
                self.early_stop = True
                return False
            return 'n'
        else:
            self.best_score = score
            self.counter = 0
            return True



# experiments/FEMNIST/adaptive.py -a -i -s 0 -e
class FedMapServer(ABC):
    def __init__(self, config, args, model, seed, optimizer_class: Type, optimizer_params: dict,
                 use_adaptive, use_evaluate=True, lr_scheduler_class=None, lr_scheduler_params=None, control=None,
                 control_scheduler=None, resume=False, init_time_offset=0, device=None):
        self.config = config

        self.experiment_name = args.experiment_name
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

        if self.use_adaptive:
            print("Init max dec = {}. "  # 0.3
                  "Adjustment dec half-life = {}. "  # 10000
                  "Adjustment interval = {}.".format(self.config.MAX_DEC_DIFF, self.config.ADJ_HALF_LIFE,
                                                     self.config.ADJ_INTERVAL))  # 50

        self.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)

        mkdir_save(self.model, os.path.join(self.save_path, "init_model.pt"))

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

        self.use_coeff = False if args.uc == 'n' else True
        self.display = False
        self.logger = None
        self.old_list_mask = None
        self.old_accumulate_weight_dict = None
        self.list_loss = []
        self.list_acc = []
        self.fed_avg_acc = []
        self.fed_avg_loss = []
        self.list_est_time = []
        self.list_model_size = []
        self.list_client_size = []
        self.list_client_loss = []
        self.list_client_acc = []
        self.sever_to_client_sum = []
        self.list_l1_client_weight_sum = []
        self.list_l2_client_weight_sum = []
        self.list_l1_client_bias_sum = []
        self.list_l2_client_bias_sum = []
        self.lr_mask_dict = {}

        self.model_size = []
        self.list_optimizer = None
        self.list_lr_scheduler = None
        self.early_stop = False
        self.list_client_density = []
        self.early_stoping = EarlyStopping(patience=self.config.patience)
        self.old_mask = None
        self.new_mask = None
        self.lr_warm = False
        self.warming_up = False
        self.list_sum_mask = []

    def save_checkpoint(self, list_optimizer, list_lr_scheduler):
        '''
            功能：当验证损失减少时保存模型
            input:
                val_loss         验证损失
                model            模型
                model_path       模型保存地址
        '''
        self.list_est_time.append(timer() - self.start_time)
        self.list_optimizer = list_optimizer
        self.list_lr_scheduler = list_lr_scheduler
        checkpoint = {"self.sever_to_client_sum": self.sever_to_client_sum,
                      "self.list_loss": self.list_loss,
                      "self.list_acc": self.list_acc,
                      'self.fed_avg_loss': self.fed_avg_loss,
                      'self.lr_mask_dict': self.lr_mask_dict,
                      'self.fed_avg_acc': self.fed_avg_acc,
                      'self.list_est_time': self.list_est_time,
                      'self.model': self.model,
                      'self.start_time': self.start_time,
                      'self.list_client_acc': self.list_client_acc,
                      'self.list_client_loss': self.list_client_loss,
                      'self.list_l1_client_weight_sum': self.list_l1_client_weight_sum,
                      'self.list_l2_client_weight_sum': self.list_l2_client_weight_sum,
                      'self.list_l1_client_bias_sum': self.list_l1_client_bias_sum,
                      'self.list_l2_client_bias_sum': self.list_l2_client_bias_sum,
                      'self.model_G': self.model_G,
                      'self.model_idx': self.model_idx,
                      'self.control.accumulate_weight_dict': self.control.accumulate_weight_dict,
                      'self.list_mask': self.list_mask,
                      'self.list_model_size': self.list_model_size,
                      'self.list_client_density': self.list_client_density,
                      'self.list_optimizer': self.list_optimizer,
                      'self.list_lr_scheduler': self.list_lr_scheduler}
        checkpoint_path = os.path.join(self.save_path, 'checkpoint.pth')
        mkdir_save(self.list_client_acc, os.path.join(self.save_path, 'list_client_acc.pt'))
        mkdir_save(self.fed_avg_acc, os.path.join(self.save_path, 'fed_avg_acc.pt'))
        mkdir_save(self.list_client_loss, os.path.join(self.save_path, 'list_client_loss.pt'))
        mkdir_save(self.fed_avg_loss, os.path.join(self.save_path, 'fed_avg_loss.pt'))
        mkdir_save(self.model_G, os.path.join(self.save_path, 'model_G.pt'))
        mkdir_save(self.list_client_density, os.path.join(self.save_path, 'self.list_client_density'))
        mkdir_save(checkpoint, checkpoint_path)


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

            # self.model_size = load(os.path.join(self.save_path, "model_size.pt"))
            # self.list_loss = load(os.path.join(self.save_path, "loss.pt"))
            # self.list_acc = load(os.path.join(self.save_path, "accuracy.pt"))
            # self.fed_avg_loss = load(os.path.join(self.save_path, "fed_avg_loss.pt"))
            # self.fed_avg_acc = load(os.path.join(self.save_path, "fed_avg_accuracy.pt"))
            # self.list_est_time = load(os.path.join(self.save_path, "est_time.pt"))
            # self.model = load(os.path.join(self.save_path, "model.pt"))
            # self.list_model_size = load(os.path.join(self.save_path, "list_model_siz.pt"))
            # self.list_client_acc = load(os.path.join(self.save_path, "list_client_acc.pt"))
            # self.list_client_loss = load(os.path.join(self.save_path, "list_client_loss.pt"))
            # self.list_client_sum = load(os.path.join(self.save_path, "list_model_sum.pt"))
            # self.sever_to_client_sum = load(os.path.join(self.save_path, "sever_to_client_sum.pt"))
            # self.R2SP_client_sum = load(os.path.join(self.save_path, "R2SP_client_sum.pt"))
            # self.list_time_stamp = load(os.path.join(self.save_path, "time.pt"))
            # accumulate_weight_dict = load(os.path.join(self.save_path, 'accumulate_weight_dict.pt'))
            # old_accumulate_weight_dict = load(os.path.join(self.save_path, 'old_accumulate_weight_dict.pt'))
            # self.model_G = load(os.path.join(self.save_path, 'model_G.pt'))
            # start_time = load(os.path.join(self.save_path, 'start_time.pt'))
            # old_model_idx = load(os.path.join(self.save_path, 'old_model_idx.pt'))
            # model_idx = load(os.path.join(self.save_path, 'model_idx.pt'))
            # load(os.path.join(self.save_path, 'model_G.pt'))
            # len_model_G = len(self.model_G)
            len_model_size = len(self.fed_avg_acc)

            self.round = (len_model_size - 1) * self.config.EVAL_DISP_INTERVAL

            self.start_time = timer() - self.list_est_time[-1]
            self.model = self.model.to(self.device)
            self.control.model = self.model
            list_state_dict, sum_time = self.split_model(timer())
            print('currend_idx: '+str(self.round))
            print("Server resumed")
            model_state_dict = []
            for i in range(len(client_list)):
                client_model_idx = self.model_idx[i][-1]
                model_state_dict.append(list_state_dict[client_model_idx])

                # for key in state_dict_key:
                #     if model_state_dict[i][key].is_sparse:
                #         model_state_dict[i][key] = model_state_dict[i][key].to_dense()
            # I think maybe need
            remaining_batches = (self.round + 1) * self.exp_config.num_local_updates * len(client_list)
            used_sampler = client_list[0].train_loader.sampler

            num_batches_epoch = len(client_list[0].train_loader)
            print(len(used_sampler))
            used_sampler.sequence = used_sampler.sequence[remaining_batches:]
            print(len(used_sampler))
            while remaining_batches >= num_batches_epoch:
                remaining_batches -= num_batches_epoch
                print('dataloader skip epoch')
            from torch.utils.data import DataLoader, SubsetRandomSampler
            from bases.vision.load import get_data_loader
            train_data_loader = get_data_loader(self.config.EXP_NAME,data_type="train", batch_size=client_list[0].train_loader.batch_size,
                                            sampler=used_sampler, num_workers=self.config.train_num, pin_memory=True)

            for i in range(len(client_list)):
                client = client_list[i]
                client.optimizer_wrapper.optimizer.load_state_dict(self.list_optimizer[i])
                if client.optimizer_wrapper.lr_scheduler is not None: client.optimizer_wrapper.lr_scheduler.load_state_dict(self.list_lr_scheduler[i])
                client.load_state_dict(model_state_dict[i])
                client.optimizer_wrapper.lr_scheduler_step()
                client.init_train_loader(train_data_loader)




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

    def get_real_size(self, list_state_dict,exp):
        list_model_size = []
        for client_model in list_state_dict:
            with open((exp+'.pkl'), 'wb') as f:
                pickle.dump(client_model, f)
            file_size = os.path.getsize(exp+'.pkl')
            file_size = file_size / (1024 * 1024)
            list_model_size.append(file_size)
            os.remove(exp+'.pkl')
        return list_model_size
    def clean_dict_to_client(self, state_dict) -> dict:
        """
        Clean up state dict before processing, e.g. remove entries, transpose.
        To be overridden by subclasses.
        """
        with torch.no_grad():
            clean_state_dict = copy_dict(state_dict)  # not deepcopy

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
        clean_time = []
        with torch.no_grad():
            for i in range(len(list_state_dict)):
                clean_state_dict_time = timer()
                list_state_dict[i] = self.clean_dict_to_client(list_state_dict[i])
                clean_time.append(timer() - clean_state_dict_time)

            process_dict_time = []
            for clean_state_dict in list_state_dict:
                process_time = timer()
                for layer, prefix in zip(self.model.prunable_layers, self.model.prunable_layer_prefixes):
                    # works for both layers
                    key_w = prefix + ".weight"
                    if key_w in clean_state_dict.keys():
                        weight = clean_state_dict[key_w]
                        sparse_weight = weight.view(weight.size(0), -1).to_sparse()
                        if sparse_weight._nnz() == 0:
                            sparse_weight = None
                        clean_state_dict[key_w] = sparse_weight
                        process_dict_time.append(timer() - process_time)

        return list_state_dict, clean_time, process_dict_time

    def process_state_dict_to_client_fast(self, list_state_dict):
        for i in range(len(list_state_dict)):
            list_state_dict[i] = self.clean_dict_to_client(list_state_dict[i])
        return list_state_dict

    @torch.no_grad()
    def merge_accumulate_client_update(self, list_num_proc, list_state_dict,
                                       idx):  # to complete the merge model ps: fedavg
        if self.display:
            self.logger.info('use sub_fedavg_and_fair to merge client')
        # merged_state_dict = dict()
        dict_keys = list_state_dict[0].keys()
        for state_dict in list_state_dict[1:]:
            assert state_dict.keys() == dict_keys
        # to check that all the state_dict have the same structure
        # sub_fed_avg
        if self.display:
            self.logger.info('list_num_proc'+str(list_num_proc))

        count = {}
        sd = copy.deepcopy(self.model.state_dict())
        for key in dict_keys:
            sum_weight = torch.zeros(size=list_state_dict[0][key].size(),device=self.device)
            sum_mask = torch.zeros(size=sum_weight.size(), device=self.device)
            mask = None
            for num_proc, state_dict in zip(list_num_proc, list_state_dict):
                if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense()
                sum_weight = sum_weight + num_proc * state_dict[key].to(self.device)
                mask = (state_dict[key] != 0).to(self.device)
                sum_mask = sum_mask + mask * num_proc

            divisor = torch.where(sum_mask == 0, torch.tensor([1e-10], device=self.device), sum_mask)

            sum_weight = torch.div(sum_weight, divisor)
            # for num_proc, state_dict in zip(list_num_proc, list_state_dict):
            #     sum_weight = sum_weight + num_proc / total_num_proc * state_dict[key].to_dense()

            sd[key] = sum_weight.view(sd[key].size())
            #self.control.accumulate(key, idx)
        old_model = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(sd)
        for key in dict_keys:
            if self.accumulate == 'w':
                self.control.accumulate(key, idx, max(self.interval,10))
            elif self.accumulate == 'g':
                self.control.accumulate_wg(old_model,key,idx,max(self.interval,10))
            else:
                if self.display:
                    self.logger.info('wrong accumulate')


    @torch.no_grad()
    def merge_accumulate_client_update_test(self, list_num_proc, list_state_dict,
                                       idx):  # to complete the merge model ps: fedavg
        if self.display:
            self.logger.info('use sub_fedavg_and_fair to merge client')
        # merged_state_dict = dict()
        dict_keys = self.model.state_dict().keys()
        # for state_dict in list_state_dict[1:]:
        #     assert state_dict.keys() == dict_keys
        # to check that all the state_dict have the same structure
        # sub_fed_avg
        if self.display:
            self.logger.info('list_num_proc'+str(list_num_proc))

        count = {}
        sd = copy.deepcopy(self.model.state_dict())
        for key in dict_keys:
            sum_weight = torch.zeros(size=list_state_dict[0][key].size(), device=self.device)
            sum_mask = torch.zeros(size=sum_weight.size(), device=self.device)
            mask = None
            i = 0
            for num_proc, state_dict in zip(list_num_proc, list_state_dict):
                if i < 7:
                    if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense().to(self.device)
                    sum_weight = sum_weight + num_proc * state_dict[key]
                    mask = (state_dict[key] != 0).to(self.device)
                    sum_mask = sum_mask + mask * num_proc
                else:
                    pass
                i = i+1


            divisor = torch.where(sum_mask == 0, torch.tensor([1e-10]).to(self.device), sum_mask)

            sum_weight = torch.div(sum_weight, divisor).to(self.device)
            # for num_proc, state_dict in zip(list_num_proc, list_state_dict):
            #     sum_weight = sum_weight + num_proc / total_num_proc * state_dict[key].to_dense()

            sd[key] = sum_weight.view(sd[key].size())
            #self.control.accumulate(key, idx)

        self.model.load_state_dict(sd)
        for (key, param) in self.model.named_parameters():
            self.control.accumulate(key, idx, max(self.interval,10))

    @torch.no_grad()
    def R2SP(self, list_num_proc, list_state_dict, idx):  # to complete the merge model ps: fedavg
        if self.display:
            self.logger.info('use R2SP to merge client')
        # merged_staSPte_dict = dict()
        dict_keys = list_state_dict[0].keys()
        for state_dict in list_state_dict[1:]:
            assert state_dict.keys() == dict_keys
        # to check that all the state_dict have the same structure
        # sub_fed_avg
        if self.display:
            self.logger.info('list_num_proc'+str(list_num_proc))


        for state_dict in list_state_dict:
            for key in dict_keys:
                if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense().view(self.model.state_dict()[key].size())
                state_dict[key] = state_dict[key].view(self.model.state_dict()[key].size())
                state_dict[key][state_dict[key] == 0] = self.model.state_dict()[key][state_dict[key] == 0]

        total_num_proc = sum(list_num_proc)

        with torch.no_grad():
            for key, param in self.model.state_dict().items():
                avg_inc_val = None
                for num_proc, state_dict in zip(list_num_proc, list_state_dict):
                    if key in state_dict.keys():

                        mask = state_dict[key] != 0
                        if mask is None:
                            inc_val = state_dict[key].to_dense() - param
                        else:
                            inc_val = state_dict[key].to_dense() - param
                        inc_val.view(param.size())

                        if avg_inc_val is None:
                            avg_inc_val = num_proc / total_num_proc * inc_val
                        else:
                            avg_inc_val += num_proc / total_num_proc * inc_val

                if avg_inc_val is None or key.endswith("num_batches_tracked"):
                    continue
                else:
                    param.add_(avg_inc_val)
        dict_keys = list_state_dict[0].keys()
        for key in dict_keys:
            self.control.accumulate(key, idx, max(self.interval, 10))

    @torch.no_grad()
    def fed_avg_model(self, list_num_proc, list_state_dict, idx):
        total_num_proc = sum(list_num_proc)

        model_state_dict = copy.deepcopy(self.model).to('cpu').state_dict()

        with torch.no_grad():
            for key, param in self.model.state_dict().items():
                avg_inc_val = None
                for num_proc, state_dict in zip(list_num_proc, list_state_dict):
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
                            avg_inc_val = num_proc / total_num_proc * inc_val
                        else:
                            avg_inc_val += num_proc / total_num_proc * inc_val

                if avg_inc_val is None or key.endswith("num_batches_tracked"):
                    continue
                else:
                    param.add_(avg_inc_val)
        fed_avg_model = copy.deepcopy(self.model)
        self.model.load_state_dict(model_state_dict)
        return fed_avg_model

    @torch.no_grad()
    def fed_avg(self, list_num_proc, list_state_dict, idx):  # to complete the merge model ps: fedavg
        if self.display:
            self.logger.info('use fedavg to merge client')
        total_num_proc = sum(list_num_proc)

        with torch.no_grad():
            for key, param in self.model.state_dict().items():
                avg_inc_val = None
                for num_proc, state_dict in zip(list_num_proc, list_state_dict):
                    if key in state_dict.keys():
                        state_dict[key] = state_dict[key].to_dense().view(param.size())
                        mask = state_dict[key].to_dense() != 0
                        if mask is None:
                            inc_val = state_dict[key].to_dense() - param
                        else:
                            inc_val = state_dict[key].to_dense() - param
                        inc_val.view(param.size())

                        if avg_inc_val is None:
                            avg_inc_val = num_proc / total_num_proc * inc_val
                        else:
                            avg_inc_val += num_proc / total_num_proc * inc_val

                if avg_inc_val is None or key.endswith("num_batches_tracked"):
                    continue
                else:
                    param.add_(avg_inc_val)
        dict_keys = list_state_dict[0].keys()
        for key in dict_keys:
            self.control.accumulate(key, idx, max(self.interval, 10))
    @torch.no_grad()
    def load_client_dict(self, client_dict, model):  # to complete the merge model ps: fedavg
        if self.display:
            self.logger.info('use fedavg to merge client')
        client_state_dict = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            for key, param in client_state_dict.items():
                if key in client_dict.keys():
                    if client_dict[key].is_sparse: client_dict[key] = client_dict[key].to_dense()
                    client_state_dict[key] = client_dict[key].view(param.size())

        model.load_state_dict(client_state_dict)
        return model


    @torch.no_grad()
    def split_model(self,start):

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
            clean_model.to('cpu')
            clean_state_dict = clean_model.state_dict()
            list_state_dict.append(clean_state_dict)

            sub_model_time.append(timer()-start)
        return list_state_dict, sub_model_time


    def main(self, idx, list_sd, list_num_proc, lr, list_client_loss,list_client_acc, list_client_sum, start, list_loss, list_acc, list_est_time,
             list_model_size, is_adj_round, client_density, list_optimizer,list_lr_scheduler):



        if idx % self.config.EVAL_DISP_INTERVAL == 0:
            self.display = True
        else:
            self.display = False
        self.round = idx

        sub_fedavg_time_start = timer()

        if self.merge == 'fed_avg':
            self.fed_avg(list_num_proc, list_sd, idx)
        elif self.merge == 'sub_fed_avg':
            self.merge_accumulate_client_update(list_num_proc, list_sd, idx)
        elif self.merge == 'sub_fed_avg_test':
            self.merge_accumulate_client_update_test(list_num_proc, list_sd, idx)
        elif self.merge == 'R2SP':
            self.R2SP(list_num_proc, list_sd, idx)
         # to complete the merge model ps: fedavg
        sub_fedavg_time = timer() - sub_fedavg_time_start

        self.logger.info('sub_fedavg_time =  '+str(sub_fedavg_time))


        sum_g = 0
        for key in self.control.g.keys():
            sum_g += torch.abs(self.control.g[key]**2).sum()
        self.model_G.append(sum_g.cpu())
        if idx % self.config.EVAL_DISP_INTERVAL == 0:

            # loss, acc = self.model.evaluate(self.test_loader)
            # list_loss.append(loss)
            # list_acc.append(acc)
            fed_avg_model = self.fed_avg_model(list_num_proc, list_sd, idx)
            client_acc = []
            client_loss = []

            subset_data_loader = self.test_loader


            # for i in range(len(list_sd)+1):
            for i in range( 1):
                if i == 0:
                    if self.config.EXP_NAME == "TinyImageNet":
                        from torch.utils.data import DataLoader, SubsetRandomSampler
                        # 计算要抽样的子集大小（假设是原数据集大小的 1/10）
                        subset_size = len(self.test_loader.dataset) // 3
                        # 生成随机的子集索引
                        indices = torch.randperm(len(self.test_loader.dataset))[:subset_size]
                        # 使用 SubsetRandomSampler 创建新的数据加载器
                        subset_sampler = SubsetRandomSampler(indices)

                        subset_data_loader = DataLoader(dataset=self.test_loader.dataset,
                                                        batch_size=self.test_loader.batch_size,
                                                        sampler=subset_sampler, num_workers=self.config.test_num)
                    fed_avg_model.to(self.device)
                    avg_loss, avg_acc = fed_avg_model.evaluate(subset_data_loader)
                    self.fed_avg_acc.append(avg_acc)
                    self.fed_avg_loss.append(avg_loss)
                # else:
                #     if self.config.EXP_NAME == "TinyImageNet" or self.config.EXP_NAME == "CelebA":
                #         if self.config.EXP_NAME == "TinyImageNet": n =10
                #         if self.config.EXP_NAME == "CelebA": n = 5
                #         from torch.utils.data import DataLoader, SubsetRandomSampler
                #         # 计算要抽样的子集大小（假设是原数据集大小的 1/10）
                #         subset_size = len(self.test_loader.dataset) // n
                #         # 生成随机的子集索引
                #         indices = torch.randperm(len(self.test_loader.dataset))[:subset_size]
                #         # 使用 SubsetRandomSampler 创建新的数据加载器
                #         subset_sampler = SubsetRandomSampler(indices)
                #
                #         subset_data_loader = DataLoader(dataset=self.test_loader.dataset,
                #                                         batch_size=self.test_loader.batch_size,
                #                                         sampler=subset_sampler, num_workers=self.config.test_num)
                #     fed_avg_model = self.load_client_dict(list_sd[i-1], fed_avg_model)
                #     avg_loss, avg_acc = fed_avg_model.evaluate(subset_data_loader)
                #     client_acc.append(avg_acc)
                #     client_loss.append(avg_loss)

            self.list_client_loss.append(client_loss)
            self.list_client_acc.append(client_acc)

            if self.display:
                self.logger.info("Round #{} (Experiment = {}).".format(idx, self.experiment_name))
                # self.logger.info("Loss/acc (at round #{}) = {}/{}".format(self.round, avg_loss, avg_acc))
                print("fed_avg Loss/acc (at round #{}) = {}/{}".format(self.round, self.fed_avg_loss[-1], self.fed_avg_acc[-1]))
                self.logger.info("fed_avg Loss/acc (at round #{}) = {}/{}".format(self.round, self.fed_avg_loss[-1], self.fed_avg_acc[-1]))
        self.logger.info("Elapsed time = {}".format(timer() - self.start_time))



        # to adjust model or split model
        if idx % self.interval == 0:

            if self.display:
                self.logger.info("Running Sub_pruning algorithm")
            list_state_dict, model_idx, sub_model_time, list_mask, self.list_sum_mask = self.control.sub_adjust_fast(
                client_density=client_density,use_coff=self.use_coeff,
                min_density=0.001,)
            self.old_list_mask = list_mask
            self.list_mask = list_mask
            self.old_model_idx = model_idx
            self.model_idx = model_idx
            self.logger.info('sub_adjust_time =  ' + str(timer() - sub_fedavg_time_start))


        else:
            start = timer()
            if self.display:
                self.logger.info('interval round, use mask to split model')
            list_state_dict, sub_model_time = self.split_model(start)
            model_idx = self.model_idx
            self.logger.info('sub_split_time =  ' + str(timer() - sub_fedavg_time_start))

        # list_state_dict, clean_time, process_dict_time = self.process_state_dict_to_client(list_state_dict)
        list_state_dict = self.process_state_dict_to_client_fast(list_state_dict)

        client_l1_weight_sum = []
        client_l2_weight_sum = []
        client_l1_bias_sum = []
        client_l2_bias_sum = []
        mask_sum = 0
        for sum_mask in self.list_sum_mask:
            l1_weight_sum = 0
            l2_weight_sum = 0

            num = 0
            for layer, prefix in zip(self.model.prunable_layers, self.model.prunable_layer_prefixes):
                # works for both layers
                key_w = prefix + ".weight"
                if key_w in sum_mask.keys():
                    weight = torch.abs(self.model.state_dict()[key_w] * sum_mask[key_w]).sum()
                    l1_weight_sum += torch.abs(weight).sum()
                    l2_weight_sum += (weight**2).sum()
                    num += torch.sum(sum_mask[key_w]).item()

            client_l1_weight_sum.append(copy.copy(l1_weight_sum.cpu()))
            client_l2_weight_sum.append(copy.copy(l2_weight_sum.cpu()))
            mask_sum = num
        for i in range(len(client_l1_weight_sum)):
            c = len(client_l1_weight_sum)-1-i
            if c != 0:
                client_l1_weight_sum[c] = client_l1_weight_sum[c] - client_l1_weight_sum[c-1]
                client_l2_weight_sum[c] = client_l2_weight_sum[c] - client_l2_weight_sum[c - 1]


        self.list_l1_client_weight_sum.append(client_l1_weight_sum)
        self.list_l2_client_weight_sum.append(client_l2_weight_sum)





        if self.display:
            self.logger.info('client_l1_weight_sum: '+str(mask_sum/10)+ ' '+str(client_l1_weight_sum) )
            self.logger.info('client_l2_weight_sum: ' + str(client_l2_weight_sum) )

            print('client_l1_weight_sum: '+str(mask_sum/10)+ ' '+str(client_l1_weight_sum) )
            print('client_l2_weight_sum: ' + str(client_l2_weight_sum) )


        # model_state_dict = []
        # state_dict_key = list_state_dict[0].keys()
        #
        # for i in range(len(list_state_dict)):
        #     client_model_idx = model_idx[i][-1]
        #
        #     model_state_dict.append(list_state_dict[client_model_idx])
        #     for key in state_dict_key:
        #         if model_state_dict[i][key].is_sparse:
        #             model_state_dict[i][key] = model_state_dict[i][key].to_dense()

        # server_to_client_sum = []
        # for sd in model_state_dict:
        #     s = 0
        #     for key in sd.keys():
        #         if sd[key].is_sparse:  sd[key] = sd[key].to_dense()
        #         s += (sd[key]**2).sum()
        #     server_to_client_sum.append(s.cpu())
        #
        # self.sever_to_client_sum.append(server_to_client_sum)

        if self.round % self.config.EVAL_DISP_INTERVAL == 0:
            c = self.early_stoping(self.fed_avg_acc[-1], self.logger)
            if c:
                self.save_checkpoint(list_optimizer=list_optimizer, list_lr_scheduler=list_lr_scheduler)
            elif c == 'n':
                self.logger.info('it is not the best model')
                pass
            else:
                    self.early_stop = True
        client_l1_weight_sum = []
        client_l2_weight_sum = []
        client_l1_bias_sum = []
        client_l2_bias_sum = []

        # for state_dict in list_state_dict:
        #     l1_weight_sum = 0
        #     l2_weight_sum = 0
        #     l1_bias_sum = 0
        #     l2_bias_sum = 0
        #     num = 0
        #     for layer, prefix in zip(self.model.prunable_layers, self.model.prunable_layer_prefixes):
        #         # works for both layers
        #         key_w = prefix + ".weight"
        #         if key_w in state_dict.keys():
        #             weight = state_dict[key_w]
        #             l1_weight_sum += torch.abs(weight).sum()
        #             l2_weight_sum += (weight**2).sum()
        #         key_w = prefix + '.bias'
        #         if key_w in state_dict.keys():
        #             weight = state_dict[key_w]
        #             l1_bias_sum += torch.abs(weight).sum()
        #             l2_bias_sum += (weight**2).sum()
        #     client_l1_weight_sum.append(copy.copy(l1_weight_sum.cpu()))
        #     client_l2_weight_sum.append(copy.copy(l2_weight_sum.cpu()))
        #     client_l1_bias_sum.append(copy.copy(l1_bias_sum.cpu()))
        #     client_l2_bias_sum.append(copy.copy(l2_bias_sum.cpu()))
        # self.list_l1_client_weight_sum.append(client_l1_weight_sum)
        # self.list_l2_client_weight_sum.append(client_l2_weight_sum)
        # self.list_l1_client_bias_sum.append(client_l1_bias_sum)
        # self.list_l2_client_bias_sum.append(client_l2_bias_sum)
        #
        #
        #
        # if self.display:
        #     self.logger.info('client_l1_weight_sum: '+ str(client_l1_weight_sum[0]) + '   ' + str(client_l1_weight_sum[-1]))
        #     self.logger.info('client_l2_weight_sum: ' + str(client_l2_weight_sum[0]) + '   ' + str(client_l2_weight_sum[-1]))
        #     self.logger.info('client_l1_bias_sum: '+ str(client_l1_bias_sum[0]) + '   ' + str(client_l1_bias_sum[-1]))
        #     self.logger.info('client_l2_bias_sum: ' + str(client_l2_bias_sum[0]) + '   ' + str(client_l2_bias_sum[-1]))
        #     print('client_l1_weight_sum: '+ str(client_l1_weight_sum[0]) + '   ' + str(client_l1_weight_sum[-1]))
        #     print('client_l2_weight_sum: ' + str(client_l2_weight_sum[0]) + '   ' + str(client_l2_weight_sum[-1]))
        #     print('client_l1_bias_sum: '+ str(client_l1_bias_sum[0]) + '   ' + str(client_l1_bias_sum[-1]))
        #     print('client_l2_bias_sum: ' + str(client_l2_bias_sum[0]) + '   ' + str(client_l2_bias_sum[-1]))
        # # sub_time = [sub_fedavg_time + ct + pt + st for ct, pt, st in zip(clean_time, process_dict_time, sub_model_time)]
        sub_time = None



        return list_state_dict, model_idx, sub_time, self.list_client_loss


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
    def load_state_dict(self, state_dict):

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

        # for layer, prefix in zip(sparse_model.param_layers, sparse_model.param_layer_prefixes):
        #     key = prefix + ".bias"
        #     if isinstance(layer, SparseLinear) and key in clean_state_dict.keys():
        #         clean_state_dict[key] = clean_state_dict[key].view(-1)
        #
        # del_list = []
        # del_suffix = "placeholder"
        # for key in clean_state_dict.keys():
        #     # clean_state_dict[key] = clean_state_dict[key].cpu()
        #     if key.endswith(del_suffix):
        #         del_list.append(key)
        #
        # for del_key in del_list:
        #     del clean_state_dict[del_key]

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

        # if self.is_sparse:
        #     for key, param in clean_state_dict.items():
        #         if param.is_sparse:
        #             clean_state_dict[key] = param._values()
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

    def main(self, idx, loss, logger):
        self.model.train()
        num_proc_data = 0
        data_list = []
        lr_rate = 1

        if idx-1 % self.config.EVAL_DISP_INTERVAL == 0:

            if self.args.lr_scheduler:
                self.scheduler.step(loss)

        real_density = self.calc_model_params()
        model_density =  round(real_density, 4)
        # if real_density > 0.8: model_density = 0.1
        # if real_density < 0.2: model_density = 1.0
        if self.args.control_lr != 0:
            if model_density < 1:
                if self.args.control_lr>0:
                    lr_rate = ((1 - model_density) / 2)*self.args.control_lr + 1

                if self.args.control_lr < 0:
                    lr_rate = 1/(-(self.args.control_lr)*(1-model_density)/2+1)


        for _ in range(self.config.NUM_LOCAL_UPDATES):  # the number of local update
            inputs, labels = self.train_loader.get_next_batch()
            data_list.append([inputs, labels])



        if self.args.ft == 'y':
            # fine_tune
            for param_group in self.optimizer_wrapper.optimizer.param_groups:
                param_group['lr'] = 0.001


            for i in range(self.config.NUM_LOCAL_UPDATES):
                inputs, labels = data_list[i][0], data_list[i][1]
                self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))
                # num_proc_data += len(inputs)
            # print(self.optimizer_wrapper.optimizer.param_groups[0]['lr'])
        if self.args.control_lr != 0:
            for param_group in self.optimizer_wrapper.optimizer.param_groups:
                param_group['lr'] = self.config.INIT_LR / lr_rate


        if self.args.fair_degree == 0 :
            for i in range(self.config.NUM_LOCAL_UPDATES):
                 inputs, labels = data_list[i][0], data_list[i][1]

                 self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))
                 num_proc_data += len(inputs)

        elif self.args.fair_degree > 0:
            locale_update = int(((1-model_density)/self.args.fair_degree+1)*self.config.NUM_LOCAL_UPDATES)
            for i in range(self.config.NUM_LOCAL_UPDATES):
                 inputs, labels = data_list[i][0], data_list[i][1]
                 self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))
                 num_proc_data += len(inputs)

            data_list = data_list * 4
            random.shuffle(data_list)
            if self.args.fair == 'no_fair':
                pass
            elif self.args.fair == 'n':
                for i in range(locale_update - self.config.NUM_LOCAL_UPDATES):
                    inputs, labels = data_list[i][0],data_list[i][1]
                    # self.optimizer_wrapper.step(inputs,labels)
                    num_proc_data += len(inputs)
            elif self.args.fair == 'u':
                for i in range(locale_update - self.config.NUM_LOCAL_UPDATES):
                    inputs, labels = data_list[i][0],data_list[i][1]
                    self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))
                    # num_proc_data += len(inputs)
            elif self.args.fair == 'un' or self.args.fair == 'nu':
                for i in range(locale_update - self.config.NUM_LOCAL_UPDATES):
                    inputs, labels = data_list[i][0], data_list[i][1]
                    self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))
                    num_proc_data += len(inputs)


        if self.args.fair_degree < 0:
            # if fair degree = -1, the worst model will subtract 1 from the number of local update
            local_update = max(self.config.NUM_LOCAL_UPDATES + int(self.args.fair_degree*(1-model_density)/0.79),0)
            if self.args.fair == 'no_fair':
                for i in range(self.config.NUM_LOCAL_UPDATES):
                    inputs, labels = data_list[i][0],data_list[i][1]
                    self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))
                    num_proc_data += len(inputs)


            elif self.args.fair == 'n':
                for i in range(self.config.NUM_LOCAL_UPDATES):
                    inputs, labels = data_list[i][0],data_list[i][1]
                    self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))

                for i in range(local_update):
                    inputs, labels = data_list[i][0],data_list[i][1]
                    num_proc_data += len(inputs)
            elif self.args.fair == 'u':
                for i in range(local_update):
                    inputs, labels = data_list[i][0],data_list[i][1]
                    self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))

                for i in range(self.config.NUM_LOCAL_UPDATES):
                    inputs, labels = data_list[i][0],data_list[i][1]
                    num_proc_data += len(inputs)

            elif self.args.fair == 'un' or self.args.fair == 'nu':
                for i in range(local_update):
                    inputs, labels = data_list[i][0],data_list[i][1]
                    self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))
                    num_proc_data += len(inputs)

            if self.args.fair == 'no_fair':
                pass
        sum = 0
        for key in self.model.state_dict().keys():
            sum += (self.model.state_dict()[key] ** 2).sum()
        sum = sum.cpu()

        lr = self.optimizer_wrapper.optimizer.param_groups[0]['lr']

        # sparse_model = self.model.to_sparse()
        state_dict_to_server = self.process_state_dict_to_server(self.model)

        return state_dict_to_server, num_proc_data, lr, model_density, sum


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

        self.list_loss, self.list_acc, self.list_est_time, self.list_model_size = [], [], [], []

        self.average_download_speed = [20, 20, 20, 10, 10, 10, 10, 2, 2, 2]
        self.variance_download = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        self.average_upload_speed = [5, 5, 5, 2.5, 2.5, 2.5, 2.5, 0.5, 0.5, 0.5]
        self.variance_upload = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        self.average_server_down = 10
        self.variance_server_down = 0.3
        self.average_server_up = 3
        self.variance_server_up = 0.3
        self.computerlr = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.client_density = args.client_density
        self.density_size = {}




    def get_real_size(self, list_state_dict,exp, density):
        list_model_size = []
        for i in range(len(list_state_dict)):
            ds = density[i]
            if ds in self.density_size.keys():
                list_model_size.append(self.density_size[ds])
            else:
                client_model = list_state_dict[i]
                with open((exp + '.pkl'), 'wb') as f:
                    pickle.dump(client_model, f)
                file_size = os.path.getsize(exp + '.pkl')
                file_size = file_size / (1024 * 1024)
                list_model_size.append(file_size)
                self.density_size[ds] = file_size
                os.remove(exp + '.pkl')
        return list_model_size

    def get_internet_speed(self):

        download_speed, upload_speed = [], []
        import numpy as np
        server_up = np.random.lognormal(mean=0, sigma=self.variance_server_up) * self.average_server_up
        server_down = np.random.lognormal(mean=0, sigma=self.variance_server_down) * self.average_server_down

        for i in range(len(self.average_upload_speed)):
            dp = np.random.lognormal(mean=0, sigma=self.variance_upload[i]) * self.average_download_speed[i]
            up = np.random.lognormal(mean=0, sigma=self.variance_download[i]) * self.average_upload_speed[i]
            download_speed.append(dp)
            upload_speed.append(up)

        return server_up, server_down, download_speed, upload_speed

    def main(self):
        # model initialization completed

        log_file_path = os.path.join(self.server.save_path, 'app.log')
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(log_file_path),  # 写入文件
                                # logging.StreamHandler()  # 输出到控制台
                            ])
        # 创建一个日志记录器
        logger = logging.getLogger(__name__)

        idx = 0
        self.server.logger = logger
        if self.resume:
            self.server.resume(self.resume,self.client_list)
            idx = self.server.round+1

        server_time = 0
        list_client_loss, list_client_acc, list_client_sum, list_model_size =  [], [], [], []
        start = timer()
        self.server.start_time = start
        display = False
        communicate_time_from_server_to_client = [0] * len(self.client_list)
        while idx < self.max_round:
            if idx % self.config.EVAL_DISP_INTERVAL == 0:
                display = True
            else:
                display = False
            if self.server.early_stop:
                logger.info('early stop at'+str(idx) + ' round of FL')
                mkdir_save(self.server.list_client_acc, os.path.join(self.server.save_path, 'list_client_acc.pt'))
                mkdir_save(self.server.fed_avg_acc, os.path.join(self.server.save_path, 'fed_avg_acc.pt'))
                mkdir_save(self.server.list_client_loss, os.path.join(self.server.save_path, 'list_client_loss.pt'))
                mkdir_save(self.server.fed_avg_loss, os.path.join(self.server.save_path, 'fed_avg_loss.pt'))
                mkdir_save(self.server.model_G, os.path.join(self.server.save_path, 'model_G.pt'))
                break
            logger.info(str(idx) + ' round of FL')

            list_state_dict, list_num, list_accum_sgrad, list_last_lr= [], [], [], []

            is_adj_round = False

            i = 0
            client_train_time = []
            sd_sum = []
            client_loss = [None]*len(self.client_list) if list_client_loss == [] else list_client_loss[-1]

            density = []
            client_train_realtime = timer()
            client_sum =[]
            for client in self.client_list:
                client_time = timer()
                loss = client_loss[i]
                sd, npc, last_lr, ds, sum = client.main(idx, loss,logger)
                i = i + 1
                list_state_dict.append(sd)
                list_num.append(npc)  # Amount of data for client-side training models
                list_last_lr.append(last_lr)
                density.append(ds)
                client_train_time.append((timer() - client_time) * self.computerlr[i - 1])
                client_sum.append(sum)
            list_client_sum.append(client_sum)


                # print('client_train_real_time' + str(timer()-client_train_realtime))
            logger.info('client_train_time' + str(client_train_time))
                # print(list_last_lr)

            last_lr = list_last_lr[0]
            # for client_lr in list_last_lr[1:]:
            #     assert client_lr == last_lr

            model_size = self.get_real_size(list_state_dict,self.server.experiment_name,density)
            if display:
                logger.info('client_to_server_model_size  :'+str(model_size))

            self.server.list_model_size.append(model_size)


            # server_up, server_down, download_speed, upload_speed = self.get_internet_speed()

            from control.sub_algorithm import simulate_client_to_server, determine_density, simluate_server_to_client

            # time = [ct + comt for ct, comt in zip(client_train_time, communicate_time_from_server_to_client)]

            # time = [0,1,2,3,4,5,6,7,8,9]
            #server_receive_time = simulate_client_to_server(time, list_model_size, upload_speed, server_down)
            #print('server_receive_time:   ' + str(server_receive_time))

            #client_density = determine_density(server_receive_time)
            client_density = copy.deepcopy(self.client_density)
            if self.increase == 'bts':
                min_density = 1-(2/(1+np.exp(-idx*4/self.config.MAX_ROUND))-1)*(1-self.min_density)
                # min_density = 0.1+2 / (1 + np.exp(-(idx-20000)*4/20000))*0.9
                for i in range(len(client_density)):
                    if client_density[i] < min_density:
                        client_density[i] = min_density
            elif self.increase == 'stb':
                #min_density = 1-(2/(1+np.exp(-idx*4/20000))-1)*0.9
                min_density = self.min_density+2 / (1 + np.exp(-(idx-self.config.MAX_ROUND)*4/self.config.MAX_ROUND))*(1-self.min_density)
                for i in range(len(client_density)):
                    if client_density[i] < min_density:
                        client_density[i] = min_density
            else:
                pass
            if display:
                logger.info('To determine client density')

            server_time = 0
            list_optimizer = []
            list_lr_scheduler =[]
            if idx % self.config.EVAL_DISP_INTERVAL==0:
                for client in self.client_list:
                    list_optimizer.append(client.optimizer_wrapper.optimizer.state_dict())
                    if client.optimizer_wrapper.lr_scheduler is not None:
                        list_lr_scheduler.append(client.optimizer_wrapper.lr_scheduler.state_dict())
            list_state_dict, model_idx, sub_time,self.client_loss = self.server.main(idx, list_state_dict, list_num, last_lr, list_client_loss, list_client_acc,
                                                                    list_client_sum, server_time,
                                                                    self.list_loss, self.list_acc, self.list_est_time,
                                                                    self.list_model_size, is_adj_round, client_density,list_optimizer,list_lr_scheduler)

            # time = [server_time + st for st in sub_time]

            if display:
                logger.info("begin merge split model for every client")
            # Merging split models for every client
            model_state_dict = []
            state_dict_key = list_state_dict[0].keys()
            client_MERGE_realtime = timer()
            for i in range(len(self.client_list)):
                client_model_idx = model_idx[i][-1]

                model_state_dict.append(list_state_dict[client_model_idx])
                # for key in state_dict_key:
                #     if model_state_dict[i][key].is_sparse:
                #         model_state_dict[i][key] = model_state_dict[i][key].to_dense()



            # I think maybe need
            for i in range(len(self.client_list)):
                client = self.client_list[i]
                client.load_state_dict(model_state_dict[i])
                client.optimizer_wrapper.lr_scheduler_step()
            idx += 1
            # print('client_load_model_time = '+str(timer()-client_MERGE_realtime))
