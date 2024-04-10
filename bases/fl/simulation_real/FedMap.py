import argparse
import os
from torchinfo import summary
from copy import deepcopy
import torch
from utils.save_load import mkdir_save
from utils.functional import disp_num_params
from timeit import default_timer as timer
from utils.functional import deepcopy_dict
from abc import ABC, abstractmethod
import  configs.InternetSpeed as internet_speed
import configs.comp_power as comp_power
from bases.nn.linear import DenseLinear, SparseLinear
from utils.functional import copy_dict
from typing import Union, Type, List
import os
from copy import deepcopy
from typing import Union, Type, List
from threading import Thread

import torch
from bases.fl.sockets import ServerSocket
from utils.save_load import mkdir_save, load
from bases.fl.messages import ServerToClientUpdateMessage, ServerToClientInitMessage
from abc import ABC, abstractmethod
from timeit import default_timer as timer

from bases.fl.sockets import ClientSocket
from bases.fl.messages import ClientToServerUpdateMessage
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
                 use_adaptive, use_evaluate=True, lr_scheduler_class=None, lr_scheduler_params=None, control=None,
                 control_scheduler=None, resume=False, init_time_offset=0,device = None):
        self.config = config
        self.device = device
        self.experiment_name = args.experiment_name
        self.save_path = os.path.join("results", config.EXP_NAME, args.experiment_name)
        self.save_interval = 50
        self.use_adaptive = args.use_adaptive
        self.client_selection = args.client_selection
        self.download_speed = internet_speed.high_download_speed
        self.upload_speed = internet_speed.high_upload_speed
        self.exp_config = ExpConfig(self.config.EXP_NAME, self.save_path, seed, self.config.CLIENT_BATCH_SIZE,
                               self.config.NUM_LOCAL_UPDATES, optimizer_class, optimizer_params, lr_scheduler_class,
                               lr_scheduler_params, use_adaptive, device)

        if self.use_adaptive:
            print("Init max dec = {}. "#0.3
                  "Adjustment dec half-life = {}. "#10000
                  "Adjustment interval = {}.".format(self.config.MAX_DEC_DIFF, self.config.ADJ_HALF_LIFE,
                                                     self.config.ADJ_INTERVAL))#50

        self.model = model
        model.to(device)

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
        self.list_extra_params = self.get_init_extra_params()


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

    def check_client_to_sparse(self):#if model.density() <= config.TO_SPARSE_THR ,set the model to sparse
        if not self.client_is_sparse and self.model.density() <= self.config.TO_SPARSE_THR:
            self.client_is_sparse = True

    def is_one_before_adj_round(self) -> bool:
        return self.is_adj_round(self.round + 1)

    def is_adj_round(self, rd=None) -> bool:
        if rd is None:
            rd = self.round
        return self.use_adaptive and rd > 0 and rd % self.config.ADJ_INTERVAL == 0

    def initial_pruning(self, list_est_time, list_loss, list_acc, list_model_size):
        svdata, pvdata = self.ip_train_loader.len_data, self.config.IP_DATA_BATCH * self.config.CLIENT_BATCH_SIZE
        assert svdata >= pvdata, "server data ({}) < required data ({})".format(svdata, pvdata)
        server_inputs, server_outputs = [], []
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for _ in range(self.config.IP_DATA_BATCH):
            inp, out = self.ip_train_loader.get_next_batch()
            server_inputs.append(inp.to(dev))
            server_outputs.append(out.to(dev))

        prev_density = None
        prev_num = 5
        prev_ind = []
        start = timer()
        ip_start_adj_round = None

        for server_i in range(1, self.config.IP_MAX_ROUNDS + 1):
            model_size = self.model.calc_num_all_active_params(True)
            list_est_time.append(0)
            list_model_size.append(model_size)

            if (server_i - 1) % self.config.EVAL_DISP_INTERVAL == 0:
                # test data not observable to clients, this evaluation does not happen in real systems
                loss, acc = self.model.evaluate(self.ip_test_loader)
                train_loss, train_acc = self.model.evaluate(zip(server_inputs, server_outputs))
                list_loss.append(loss)
                list_acc.append(acc)
                if ip_start_adj_round is None and train_acc >= self.config.ADJ_THR_ACC:
                    ip_start_adj_round = server_i
                    print("Start reconfiguration in initial pruning at round {}.".format(server_i - 1))
                print("Initial pruning round {}. Accuracy = {}. Loss = {}. Train accuracy = {}. Train loss = {}. "
                      "Elapsed time = {}.".format(server_i - 1, acc, loss, train_acc, train_loss, timer() - start))

            for server_inp, server_out in zip(server_inputs, server_outputs):
                list_grad = self.ip_optimizer_wrapper.step(server_inp, server_out)
                for (key, param), g in zip(self.model.named_parameters(), list_grad):
                    assert param.size() == g.size()
                    self.ip_control.accumulate(key, g ** 2)

            if ip_start_adj_round is not None and (server_i - ip_start_adj_round) % self.config.IP_ADJ_INTERVAL == 0:
                self.ip_control.adjust(self.config.MAX_DEC_DIFF)
                cur_density = disp_num_params(self.model)

                if prev_density is not None:
                    prev_ind.append(abs(cur_density / prev_density - 1) <= self.config.IP_THR)
                prev_density = cur_density

                if len(prev_ind) >= prev_num and all(prev_ind[-prev_num:]):
                    print("Early-stopping initial pruning at round {}.".format(server_i - 1))
                    del list_loss[-1]
                    del list_acc[-1]
                    break

        len_pre_rounds = len(list_acc)
        print("End initial pruning. Total rounds = {}. Total elapsed time = {}.".format(
            len_pre_rounds * self.config.EVAL_DISP_INTERVAL, timer() - start))

        return len_pre_rounds



    def clean_dict_to_client(self) -> dict:
        """
        Clean up state dict before processing, e.g. remove entries, transpose.
        To be overridden by subclasses.
        """
        clean_state_dict = copy_dict(self.model.state_dict())  # not deepcopy

        if self.client_is_sparse:
            for layer, prefix in zip(self.model.param_layers, self.model.param_layer_prefixes):
                key = prefix + ".bias"
                if isinstance(layer, DenseLinear) and key in clean_state_dict.keys():
                    clean_state_dict[key] = clean_state_dict[key].view((-1, 1))


        return clean_state_dict

    @torch.no_grad()
    def process_state_dict_to_client(self) -> dict:
        """
        Process state dict before sending to client, e.g. to cpu, to sparse, keep values only.
        if not self.client_is_sparse: send dense
        elif self.is_adj_round(): send full sparse state_dict
        else: send sparse values only
        To be overridden by subclasses.
        """
        clean_state_dict = self.clean_dict_to_client()
        if not self.client_is_sparse:
            return clean_state_dict

        if self.is_adj_round():
            for layer, prefix in zip(self.model.prunable_layers, self.model.prunable_layer_prefixes):
                # works for both layers
                key_w = prefix + ".weight"
                if key_w in clean_state_dict.keys():
                    weight = clean_state_dict[key_w]
                    w_mask = self.model.get_mask_by_name(key_w)
                    sparse_weight = (weight * w_mask).view(weight.size(0), -1).to_sparse()
                    clean_state_dict[key_w] = sparse_weight

        else:
            for prefix in self.model.prunable_layer_prefixes:
                key_w = prefix + ".weight"
                if key_w in clean_state_dict.keys():
                    clean_state_dict[key_w] = clean_state_dict[key_w].masked_select(self.model.get_mask_by_name(key_w))

        return clean_state_dict

    @torch.no_grad()
    def merge_accumulate_client_update(self, list_num_proc, list_state_dict, lr):#to complete the merge model ps: fedavg
        total_num_proc = sum(list_num_proc)

        # merged_state_dict = dict()
        dict_keys = list_state_dict[0].keys()
        for state_dict in list_state_dict[1:]:
            assert state_dict.keys() == dict_keys

        # accumulate extra sgrad and remove from state_dict
        if self.use_adaptive and self.is_adj_round():
            prefix = "extra."
            for state_dict in list_state_dict:
                del_list = []
                for key, param in state_dict.items():
                    if key[:len(prefix)] == prefix:
                        sgrad_key = key[len(prefix):]
                        mask_0 = self.model.get_mask_by_name(sgrad_key) == 0.
                        dense_sgrad = torch.zeros_like(mask_0, dtype=torch.float)
                        dense_sgrad.masked_scatter_(mask_0, param)

                        # no need to divide by lr
                        self.control.accumulate(sgrad_key, dense_sgrad)
                        del_list.append(key)

                for del_key in del_list:
                    del state_dict[del_key]

        # accumulate sgrad and update server state dict
        server_state_dict = self.model.state_dict()
        for key in dict_keys:
            server_param = server_state_dict[key]
            avg_inc_val = None
            for num_proc, state_dict in zip(list_num_proc, list_state_dict):
                if state_dict[key].size() != server_state_dict[key].size():
                    mask = self.model.get_mask_by_name(key)
                    inc_val = server_param.masked_scatter(mask, state_dict[key]) - server_param
                else:
                    inc_val = state_dict[key] - server_param

                if avg_inc_val is None:
                    avg_inc_val = num_proc / total_num_proc * inc_val
                else:
                    avg_inc_val += num_proc / total_num_proc * inc_val

                # accumulate sgrad from parameters
                if self.use_adaptive and key in dict(self.model.named_parameters()).keys():
                    self.control.accumulate(key, ((inc_val / lr) ** 2) * num_proc)

            server_param.add_(avg_inc_val)

    def main(self, idx, list_sd, list_num_proc, lr, start, list_loss, list_acc, list_est_time,
             list_model_size, is_adj_round, density_limit=None):
        total_num_proc = sum(list_num_proc)
        self.round = idx
        self.merge_accumulate_client_update(list_num_proc, list_sd,
                                            lr)  # to complete the merge model ps: fedavg



        #to check the eval round
        if idx % self.config.EVAL_DISP_INTERVAL == 0:
            loss, acc = self.model.evaluate(self.test_loader)
            list_loss.append(loss)
            list_acc.append(acc)

            print("Round #{} (Experiment = {}).".format(idx, self.experiment_name))
            print("Loss/acc (at round #{}) = {}/{}".format((len(list_loss) - 1) * self.config.EVAL_DISP_INTERVAL, loss,
                                                           acc))
            print("Estimated time = {}".format(sum(list_est_time)))
            print("Elapsed time = {}".format(timer() - start))
            print("Current lr = {}".format(lr))


        #to adjust the model
        if self.use_adaptive and is_adj_round:
            alg_start = timer()



            print("Running adaptive pruning algorithm")
            max_dec_diff = self.config.MAX_DEC_DIFF * (0.5 ** (idx / self.config.ADJ_HALF_LIFE))
            self.control.adjust(max_dec_diff, max_density=density_limit)
            print("Total alg time = {}. Max density = {}.".format(timer() - alg_start, density_limit))
            print("Num params:")
            disp_num_params(self.model)
            self.check_client_to_sparse()






        #to estimate the communication time and compute time
        est_time = self.config.TIME_CONSTANT
        for layer, comp_coeff in zip(self.model.prunable_layers, self.config.COMP_COEFFICIENTS):
            est_time += layer.num_weight * (comp_coeff + self.config.COMM_COEFFICIENT)

        model_size = self.model.calc_num_all_active_params(True)
        list_est_time.append(est_time)
        list_model_size.append(model_size)

        if idx % self.save_interval == 0:
            mkdir_save(list_loss, os.path.join(self.save_path, "loss.pt"))
            mkdir_save(list_acc, os.path.join(self.save_path, "accuracy.pt"))
            mkdir_save(list_est_time, os.path.join(self.save_path, "est_time.pt"))
            mkdir_save(list_model_size, os.path.join(self.save_path, "model_size.pt"))
            mkdir_save(self.model, os.path.join(self.save_path, "model.pt"))


        state_dict_to_client = self.process_state_dict_to_client()#To call the method of clean_dict_to_client and to_sparse if the client is sparse
        client_adj = self.is_one_before_adj_round()#to check the round is the round which before the adj round
        to_sparse = self.client_is_sparse

        return state_dict_to_client,client_adj,to_sparse




class FedMapClient:
    def __init__(self, model, config, use_adaptive, extra_params, exp_config):
        self.config = config
        self.device = exp_config.device
        self.use_adaptive = use_adaptive
        self.model = deepcopy(model).to(self.device)
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
        if self.use_adaptive:
            self.dict_extra_sgrad = dict()
            self.accum_dense_grad = dict()

        self.is_adj_round = False
        self.is_sparse = False
        self.terminate = False
        self.parse_init_extra_params(extra_params)

    @abstractmethod
    def init_optimizer(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_train_loader(self, *args, **kwargs):
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

        print("Model converted to sparse")


    @torch.no_grad()
    def load_state_dict(self, state_dict):
        param_dict = dict(self.model.named_parameters())
        buffer_dict = dict(self.model.named_buffers())
        for key, param in {**param_dict, **buffer_dict}.items():
            if key in state_dict.keys():
                if state_dict[key].size() != param.size():
                    # sparse param with value only
                    param._values().copy_(state_dict[key])
                elif state_dict[key].is_sparse:
                    # sparse param at adjustment round
                    # print(param, param.size(), state_dict[key].is_sparse, state_dict[key])
                    # param.zero_()
                    param.copy_(state_dict[key])
                    # param._indices().copy_(state_dict[key]._indices())
                    # param._values().copy_(state_dict[key]._values())
                    # need to reload mask in this case
                    param.mask.copy_(state_dict[key].mask)
                else:
                    param.copy_(state_dict[key])


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

        if self.is_sparse:
            for key, param in clean_state_dict.items():
                if param.is_sparse:
                    clean_state_dict[key] = param._values()

        if self.is_adj_round:
            clean_state_dict.update(self.dict_extra_sgrad)
            self.dict_extra_sgrad = dict()

        return clean_state_dict



    def accumulate_dense_grad_round(self):
        for key, param in self.model.named_parameters():
            if hasattr(param, "is_sparse_param"):
                if key in self.accum_dense_grad.keys():
                    self.accum_dense_grad[key] += param.dense.grad
                else:
                    self.accum_dense_grad[key] = param.dense.grad

    def accumulate_sgrad(self, num_proc_data):
        prefix = "extra."
        for key, param in self.accum_dense_grad.items():
            pkey = prefix + key
            if pkey in self.dict_extra_sgrad.keys():
                self.dict_extra_sgrad[pkey] += (param ** 2) * num_proc_data
            else:
                self.dict_extra_sgrad[pkey] = (param ** 2) * num_proc_data

            if self.is_adj_round:
                param_mask = dict(self.model.named_parameters())[key].mask == 0.
                self.dict_extra_sgrad[pkey] = self.dict_extra_sgrad[pkey].masked_select(param_mask)

    def load_mask(self, masks):
        self.list_mask = masks



    def check_client_to_sparse(self):  # if model.density() <= config.TO_SPARSE_THR ,set the model to sparse
        if not self.client_is_sparse and self.model.density() <= self.config.TO_SPARSE_THR:
            self.client_is_sparse = True


    def main(self):
        self.model.train()
        num_proc_data = 0
        for _ in range(self.config.NUM_LOCAL_UPDATES):#the number of local update
            inputs, labels = self.train_loader.get_next_batch()
            self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))

            if self.use_adaptive:
                self.accumulate_dense_grad_round()

            num_proc_data += len(inputs)

        if self.use_adaptive:
            self.accumulate_sgrad(num_proc_data)
            self.accum_dense_grad = dict()

        lr = self.optimizer_wrapper.get_last_lr()
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
        self.start_adj_round = None

    def main(self):
        len_pre_rounds = 0
        if self.use_ip:
            print("Starting initial pruning stage...")
            len_pre_rounds = self.server.initial_pruning(self.list_est_time, self.list_loss, self.list_acc,
                                                         self.list_model_size)
            print("Clients loading server model...")
            for client in self.client_list:
                client.load_state_dict(self.server.model.state_dict())
                client.load_mask([layer.mask for layer in self.server.model.prunable_layers])

        print("Starting further pruning stage...")
        start = timer()
        for idx in range(self.max_round):
            list_state_dict, list_num, list_accum_sgrad, list_last_lr = [], [], [], []
            is_adj_round = False
            if idx % self.config.EVAL_DISP_INTERVAL == 0:
                is_adj_round = self.check_adj_round(len_pre_rounds, idx)

            for client in self.client_list:
                sd, npc, last_lr = client.main()
                list_state_dict.append(sd)
                list_num.append(npc)
                #list_accum_sgrad.append(grad)
                list_last_lr.append(last_lr)
            last_lr = list_last_lr[0]
            for client_lr in list_last_lr[1:]:
                assert client_lr == last_lr

            density_limit = None
            if self.max_d is not None:
                density_limit = self.max_d
            if self.tgt_d is not None:
                assert self.tgt_d <= self.max_d
                density_limit += (self.tgt_d - self.max_d) / self.max_round * idx*10

            #to get the model from the server
            #but need to calculate the communicate time from client to server,to get the time points at which all client models were received
            communicate_time_from_client_to_server = None
            time_server_tostart = timer()
            state_dict_to_client,client_adj,to_sparse = self.server.main(idx, list_state_dict, list_num, last_lr, start,
                                                      self.list_loss, self.list_acc, self.list_est_time,
                                                      self.list_model_size, is_adj_round, density_limit)
            consume_time_servercomputer = None

            communicate_time_from_server_to_client = None#list


            for client in self.client_list:
                client.is_adj_round = client_adj
                if not client.is_sparse and to_sparse:  # if self is not sparse and the update_msg.to_sparse is true,then call the convert_to_sparse()
                    client.convert_to_sparse()  # to convert the model to sparse and update the optimzer and optimzer_wrapper
                client.load_state_dict(state_dict_to_client)
                client.optimizer_wrapper.lr_scheduler_step()

    def check_adj_round(self, pre_rounds, idx):
        if not self.use_adaptive or len(self.list_acc) == 0:
            return False
        if len(self.list_acc) * self.config.EVAL_DISP_INTERVAL \
                < pre_rounds * self.config.EVAL_DISP_INTERVAL + self.config.ADJ_INTERVAL:
            return False
        elif self.start_adj_round is None:
            self.start_adj_round = idx
            print("Starting reconfiguration at round {}.".format(idx))
            return True
        else:
            return (idx - self.start_adj_round) % self.config.ADJ_INTERVAL == 0
