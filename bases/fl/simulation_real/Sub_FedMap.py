import argparse

# Serialize the model to get the model size of different client
import pickle

from utils.functional import disp_num_params

import  configs.InternetSpeed as internet_speed

import os
from copy import deepcopy
from typing import Union, Type, List


import torch
import copy
from utils.save_load import mkdir_save, load

from abc import ABC, abstractmethod
from timeit import default_timer as timer

from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.nn.linear import DenseLinear, SparseLinear
from utils.functional import copy_dict

def parse_args():
    parser = argparse.ArgumentParser()
    mutex = parser.add_mutually_exclusive_group(required=True)
    mutex.add_argument('-a', '--adaptive',
                       help="Use adaptive pruning",
                       action='store_true',
                       dest='use_adaptive')
    mutex.add_argument('-na', '--no-adaptive',
                       help="Do not use adaptive pruning",
                       action='store_false',
                       dest='use_adaptive')

    mutex1 = parser.add_mutually_exclusive_group(required=True)
    mutex1.add_argument('-i', '--init-pruning',
                        help="Use initial pruning",
                        action='store_true',
                        dest='initial_pruning')
    mutex1.add_argument('-ni', '--no-init-pruning',
                        help="Do not use initial pruning",
                        action='store_false',
                        dest='initial_pruning')

    parser.add_argument('-c', '--client-selection',
                        help="If use client-selection",
                        action='store_true',
                        dest='client_selection',
                        default=False,
                        required=False)
    parser.add_argument('-t', '--target-density',
                        help="Target density",
                        action='store',
                        dest='target_density',
                        type=float,
                        required=False)
    parser.add_argument('-m', '--max-density',
                        help="Max density",
                        action='store',
                        dest='max_density',
                        type=float,
                        required=False)
    parser.add_argument('-s', '--seed',
                        help="The seed to use for the prototype",
                        action='store',
                        dest='seed',
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument('-e', '--exp-name',
                        help="Experiment name",
                        action='store',
                        dest='experiment_name',
                        type=str,
                        required=True)

    return parser.parse_args()

class ExpConfig:#setup the config
    def __init__(self, exp_name: str, save_dir_name: str, seed: int, batch_size: int, num_local_updates: int,
                 optimizer_class: Type, optimizer_params: dict, lr_scheduler_class: Union[Type, None],
                 lr_scheduler_params: Union[dict, None], use_adaptive: bool,device = None):
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

#experiments/FEMNIST/adaptive.py -a -i -s 0 -e
class FedMapServer(ABC):
    def __init__(self, config, args, model, seed, optimizer_class: Type, optimizer_params: dict,
                 use_adaptive, use_evaluate=True, lr_scheduler_class=None, lr_scheduler_params=None, device=None):
        self.config = config

        self.experiment_name = args.experiment_name
        self.save_path = os.path.join("results", config.EXP_NAME, args.experiment_name)
        self.save_interval = 50
        self.use_adaptive = args.use_adaptive
        self.client_selection = args.client_selection
        self.download_speed = internet_speed.high_download_speed
        self.upload_speed = internet_speed.high_upload_speed

        self.exp_config = ExpConfig(self.config.EXP_NAME, self.save_path, seed, self.config.CLIENT_BATCH_SIZE,
                               self.config.NUM_LOCAL_UPDATES, optimizer_class, optimizer_params, lr_scheduler_class,
                               lr_scheduler_params, use_adaptive)

        self.model = model
        self.model.train()

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
        self.list_extra_params = self.get_init_extra_params()


    @abstractmethod
    def get_init_extra_params(self) -> List[tuple]:
        pass

    @abstractmethod
    def init_test_loader(self):
        pass

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



    def clean_dict_to_client(self,state_dict) -> dict:
        """
        Clean up state dict before processing, e.g. remove entries, transpose.
        To be overridden by subclasses.
        """
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
    def process_state_dict_to_client(self,list_state_dict):
        """
        Process list_state_dict before sending to client, e.g. to cpu, to sparse, keep values only.
        if not self.client_is_sparse: send dense
        elif self.is_adj_round(): send full sparse state_dict
        else: send sparse values only
        To be overridden by subclasses.
        """
        clean_time = []

        for i in range(len(list_state_dict)):
            clean_state_dict_time = timer()
            list_state_dict[i] = self.clean_dict_to_client(list_state_dict[i])
            clean_time.append(timer()-clean_state_dict_time)

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
            process_dict_time.append(timer()-process_time)

        return list_state_dict, clean_time, process_dict_time


    @torch.no_grad()
    def merge_accumulate_client_update(self, list_num_proc, list_state_dict, idx):#to complete the merge model ps: fedavg
        print('use sub_fedavg_and_fair to merge client')
        total_num_proc = sum(list_num_proc)
        # merged_state_dict = dict()
        dict_keys = list_state_dict[0].keys()
        for state_dict in list_state_dict[1:]:
            assert state_dict.keys() == dict_keys
        #to check that all the state_dict have the same structure
        # sub_fed_avg

        count = {}
        sd = copy.deepcopy(self.model.state_dict())
        for key in dict_keys:
            sum_weight = torch.zeros(size=list_state_dict[0][key].size())
            sum_mask = torch.zeros(size=sum_weight.size())
            mask = None
            for num_proc, state_dict in zip(list_num_proc, list_state_dict):
                sum_weight = sum_weight + num_proc * state_dict[key].to_dense()
                mask = state_dict[key].to_dense() !=  0
                sum_mask = sum_mask + mask * num_proc

            divisor = torch.where(sum_mask == 0, torch.tensor([1e-10]), sum_mask)


            sum_weight = torch.div(sum_weight,divisor)
            # for num_proc, state_dict in zip(list_num_proc, list_state_dict):
            #     sum_weight = sum_weight + num_proc / total_num_proc * state_dict[key].to_dense()


            sd[key] = sum_weight.view(sd[key].size())
            self.control.accumulate(key, idx)


        self.model.load_state_dict(sd)
        # for key in dict_keys:
        #     self.control.accumulate(key, idx)


    @torch.no_grad()
    def fed_avg(self, list_num_proc, list_state_dict,idx):  # to complete the merge model ps: fedavg
        total_num_proc = sum(list_num_proc)

        with torch.no_grad():
            for key, param in self.model.state_dict().items():
                avg_inc_val = None
                for num_proc, state_dict in zip(list_num_proc, list_state_dict):
                    if key in state_dict.keys():
                        state_dict[key]= state_dict[key].to_dense().view(param.size())
                        mask = state_dict[key].to_dense() !=  0
                        if mask is None:
                            inc_val = state_dict[key].to_dense() - param
                        else:
                            inc_val = state_dict[key].to_dense() - param * mask
                        inc_val.view(param.size())

                        if avg_inc_val is None:
                            avg_inc_val = num_proc / total_num_proc * inc_val
                        else:
                            avg_inc_val += num_proc / total_num_proc * inc_val

                if avg_inc_val is None or key.endswith("num_batches_tracked"):
                    continue
                else:
                    param.add_(avg_inc_val)

                self.control.accumulate(key, idx)





    def main(self, idx, list_sd, list_num_proc, lr, start, list_loss, list_acc, list_est_time,
             list_model_size, is_adj_round, client_density):
        total_num_proc = sum(list_num_proc)
        self.round = idx
        print('begin sub_fed_avg')
        sub_fedavg_time_start = timer()
        self.merge_accumulate_client_update(list_num_proc, list_sd, idx)
        #self.merge_accumulate_client_update(list_num_proc, list_sd,idx)
        print(self.model.state_dict()['features.0.weight'].sum())
        #fedavg_model = self.fed_avg(list_num_proc, list_sd,idx)

        sub_fedavg_time = timer() - sub_fedavg_time_start
        # print('sub_fedavg_time =  '+str(sub_fedavg_time))



        loss, acc = self.model.evaluate(self.test_loader)
        #avg_loss, avg_acc = fedavg_model.evaluate(self.test_loader)
        #print("acc,avg_acc  " +str(acc)+str(avg_acc))
        list_loss.append(loss)
        list_acc.append(acc)
        list_est_time.append(start)
        print("Round #{} (Experiment = {}).".format(idx, self.experiment_name))
        print("Loss/acc (at round #{}) = {}/{}".format(self.round, loss, acc))

        print("Elapsed time = {}".format(timer() - self.start_time))
        #print("Current lr = {}".format(lr))
        # print('sub_print_time =  ' + str(timer()-sub_fedavg_time_start))


        print("Running Sub_pruning algorithm")
        max_dec_diff = self.config.MAX_DEC_DIFF * (0.5 ** (idx / self.config.ADJ_HALF_LIFE))

        list_state_dict, model_idx, sub_model_time = self.control.sub_adjust(client_density=client_density, min_density=0.1)
        # print('sub_adjust_time =  ' + str(timer()-sub_fedavg_time_start))



        if self.round % self.config.EVAL_DISP_INTERVAL == 0:
            mkdir_save(list_loss, os.path.join(self.save_path, "loss.pt"))
            mkdir_save(list_acc, os.path.join(self.save_path, "accuracy.pt"))
            mkdir_save(list_est_time, os.path.join(self.save_path, "est_time.pt"))
            mkdir_save(self.model, os.path.join(self.save_path, "model.pt"))

        # To call the method of clean_dict_to_client and to_sparse if the client is sparse
        list_state_dict, clean_time, process_dict_time  = self.process_state_dict_to_client(list_state_dict)

        sub_time = [sub_fedavg_time+ct+pt+st for ct, pt, st in zip(clean_time, process_dict_time, sub_model_time)]
        # print('sub_process_dict_time =  ' + str(timer() - sub_fedavg_time_start))
        return list_state_dict, model_idx, sub_time




class FedMapClient:
    def __init__(self, model, config, use_adaptive, extra_params, exp_config):
        self.config = config

        self.use_adaptive = use_adaptive
        self.model = deepcopy(model)

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
        self.parse_init_extra_params(extra_params)


    @abstractmethod
    def init_optimizer(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_train_loader(self, *args, **kwargs):
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

                    param._values().copy_(state_dict[key])
                else:
                    param.copy_(state_dict[key])
        for layer in self.model.prunable_layers:
            mask = layer.state_dict()['weight'].to_dense() != 0
            layer.mask.copy_(mask)


    @abstractmethod
    def parse_init_extra_params(self, extra_params):
        # Initialize train_loader, etc.
        pass

    def cleanup_state_dict_to_server(self) -> dict:
        """
        Clean up state dict before process, e.g. remove entries, transpose.
        To be overridden by subclasses.
        """
        clean_state_dict = copy_dict(self.model.state_dict())  # not deepcopy
        if self.is_sparse:
            for layer, prefix in zip(self.model.param_layers, self.model.param_layer_prefixes):
                key = prefix + ".bias"
                if isinstance(layer, SparseLinear) and key in clean_state_dict.keys():
                    clean_state_dict[key] = clean_state_dict[key].view(-1)

            del_list = []
            del_suffix = "placeholder"
            for key in clean_state_dict.keys():
                if key.endswith(del_suffix):
                    del_list.append(key)

            for del_key in del_list:
                del clean_state_dict[del_key]

        return clean_state_dict

    @torch.no_grad()
    def process_state_dict_to_server(self) -> dict:
        """
        Process state dict before sending to server, e.g. keep values only, extra param in adjustment round.
        if not self.is_sparse: send dense
        elif self.adjustment_round: send sparse values + extra grad values
        else: send sparse values only
        To be overridden by subclasses.
        """
        clean_state_dict = self.cleanup_state_dict_to_server()

        # if self.is_sparse:
        #     for key, param in clean_state_dict.items():
        #         if param.is_sparse:
        #             clean_state_dict[key] = param._values()
        # but in our model, different client have different density, it will have different structure

        return clean_state_dict


    def load_mask(self, masks):
        self.list_mask = masks

    def calc_model_params(self, display=False):
        sum_param_in_use = 0#the sum of all used (model layers+bias)
        sum_all_param = 0
        for layer, layer_prefix in zip(self.model.param_layers, self.model.param_layer_prefixes):
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







    def check_client_to_sparse(self):  # if model.density() <= config.TO_SPARSE_THR ,set the model to sparse
        if not self.client_is_sparse and self.model.density() <= self.config.TO_SPARSE_THR:
            self.client_is_sparse = True


    def main(self):
        self.model.train()
        num_proc_data = 0

        for _ in range(self.config.NUM_LOCAL_UPDATES):#the number of local update
            inputs, labels = self.train_loader.get_next_batch()

            self.optimizer_wrapper.step(inputs, labels)
            num_proc_data += len(inputs)
        lr = self.optimizer_wrapper.get_last_lr()
        # print('num_proc_data:  '+str(num_proc_data))

        state_dict_to_server = self.process_state_dict_to_server()

        return state_dict_to_server, num_proc_data, lr







class FedMapFL(ABC):
    def __init__(self, args, config, server, client_list):
        self.config = config
        self.use_ip = args.initial_pruning
        self.use_adaptive = args.use_adaptive
        self.tgt_d, self.max_d = args.target_density, args.max_density
        self.max_round = config.MAX_ROUND
        self.server = server
        self.client_list = client_list

        self.list_loss, self.list_acc, self.list_est_time, self.list_model_size = [], [], [], []

        self.average_download_speed = [20, 20, 20, 10, 10, 10, 10, 2, 2, 2]
        self.variance_download = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        self.average_upload_speed = [5, 5, 5, 2.5, 2.5, 2.5, 2.5, 0.5, 0.5, 0.5]
        self.variance_upload = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        self.average_server_down = 10
        self.variance_server_down = 0.3
        self.average_server_up = 3
        self.variance_server_up = 0.3
        self.computerlr = [1, 1, 1, 2, 2, 2, 2, 6, 6, 6]
        self.client_density = args.client_density

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


    def get_internet_speed(self):

        download_speed,upload_speed = [], []
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
        server_time = 0

        start = timer()
        communicate_time_from_server_to_client = [0] * len(self.client_list)
        for idx in range(self.max_round):
            print(str(idx)+' round of FL')
            list_state_dict, list_num, list_accum_sgrad, list_last_lr = [], [], [], []


            is_adj_round = False

            i = 0
            client_train_time = []
            client_train_realtime = timer()
            for client in self.client_list:
                client_time = timer()
                sd, npc, last_lr = client.main()
                i = i+1
                list_state_dict.append(sd)
                list_num.append(npc)# Amount of data for client-side training models
                #list_accum_sgrad.append(grad)
                list_last_lr.append(last_lr)
                client_train_time.append((timer()-client_time)*self.computerlr[i-1])
            # print('client_train_real_time' + str(timer()-client_train_realtime))
            print('client_train_time'+str(client_train_time))


            last_lr = list_last_lr[0]
            for client_lr in list_last_lr[1:]:
                assert client_lr == last_lr

            list_model_size = self.get_real_size(list_state_dict,self.server.experiment_name)
            #print(list_model_size)

            server_up, server_down, download_speed, upload_speed = self.get_internet_speed()



            from control.sub_algorithm import simulate_client_to_server,determine_density,simluate_server_to_client

            time = [ct+comt for ct, comt in zip(client_train_time, communicate_time_from_server_to_client)]


            #time = [0,1,2,3,4,5,6,7,8,9]
            server_receive_time = simulate_client_to_server(time,list_model_size,upload_speed,server_down)
            print('server_receive_time:   '+str(server_receive_time))

            client_density = determine_density(server_receive_time)
            client_density = self.client_density
            print('To determine client density')

            server_time = max(server_receive_time)

            list_state_dict, model_idx, sub_time = self.server.main(idx, list_state_dict, list_num, last_lr, server_time,
                                                      self.list_loss, self.list_acc, self.list_est_time,
                                                      self.list_model_size, is_adj_round, client_density)



            time = [server_time+st for st in sub_time]



            list_model_size = self.get_real_size(list_state_dict, self.server.experiment_name)
            communicate_time_from_server_to_client = simluate_server_to_client(time, list_model_size, server_up, model_idx, download_speed)

            print("begin merge split model for every client")
            #Merging split models for every client
            model_state_dict = []
            state_dict_key = list_state_dict[0].keys()
            client_MERGE_realtime = timer()
            for idx in range(len(self.client_list)):
                client_model_idx = model_idx[idx]
                dt = {}
                for sub_idx in client_model_idx:
                    for key in state_dict_key:
                        if key in dt.keys():
                            if list_state_dict[sub_idx][key] != None:

                                dt[key] = dt[key] + list_state_dict[sub_idx][key]
                        else:
                            dt[key] = list_state_dict[sub_idx][key]
                model_state_dict.append(dt)



            # I think maybe need
            for idx in range(len(self.client_list)):
                client = self.client_list[idx]
                if not client.is_sparse:  # if self is not sparse and the update_msg.to_sparse is true,then call the convert_to_sparse()
                    client.convert_to_sparse()  # to convert the model to sparse and update the optimzer and optimzer_wrapper
                client.load_state_dict(model_state_dict[idx])
                client.optimizer_wrapper.lr_scheduler_step()
            # print('client_load_model_time = '+str(timer()-client_MERGE_realtime))
