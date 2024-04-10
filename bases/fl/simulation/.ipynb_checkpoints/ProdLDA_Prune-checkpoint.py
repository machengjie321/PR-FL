import argparse
import os
from copy import deepcopy
import torch
from utils.save_load import mkdir_save
from utils.functional import disp_num_params
from timeit import default_timer as timer
from utils.functional import deepcopy_dict

from abc import ABC, abstractmethod

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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

    parser.add_argument('-r', '--rate',
                        help="learning rate",
                        action='store',
                        dest='lr',
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
                        default='ProdLDA',
                        required=False)
    parser.add_argument('-p', '--prune_rate',
                        help="Prune rate",
                        action='store',
                        dest='prune_rate',
                        type=float,
                        default=0.3,
                        required=False)


    return parser.parse_args()


class ProdLDAServer(ABC):
    def __init__(self, args, config, model, save_interval=50,data_set=None):
        self.config = config
        self.device = device
        self.experiment_name = args.experiment_name
        self.save_path = os.path.join("results", args.EXP_NAME, args.experiment_name)
        self.save_interval = config.save_interval
        self.use_adaptive = args.use_adaptive
        self.args = args
        self.data_set = data_set
        self.client_selection = args.client_selection

        if self.use_adaptive:
            print("Init target dec = {}. "#0.3
                  "Adjustment dec half-life = {}. "#10000
                  "Adjustment interval = {}.".format(self.args.target_density, self.config.ADJ_HALF_LIFE,
                                                     self.config.ADJ_INTERVAL))#50

        self.model = model.to(self.device)
        self.model.train()


        self.indices = None

        self.ip_train_loader = None
        self.ip_test_loader = None
        self.ip_optimizer_wrapper = None
        self.ip_control = None
        self.valid_data = None
        self.last_time = timer()

        self.test_loader = None
        self.control = None

        self.init_clients()
        self.init_control()

        self.save_exp_config()

    @abstractmethod
    def init_test_loader(self,x_test):
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




    def main(self, idx, list_sd, list_num_proc, lr, list_accumulated_sgrad, start, list_topic_diversity,list_npmi,list_accuracy, list_recall, list_precision, list_f1, list_loss, list_est_time,
             list_model_size, is_adj_round, density_limit=None):
        total_num_proc = sum(list_num_proc)

        with torch.no_grad():
            for key, param in self.model.state_dict().items():
                avg_inc_val = None
                for num_proc, state_dict in zip(list_num_proc, list_sd):
                    if key in state_dict.keys():
                        mask = self.model.get_mask_by_name(key)
                        if mask is None:
                            inc_val = state_dict[key] - param
                        else:
                            inc_val = state_dict[key] - param * self.model.get_mask_by_name(key)

                        if avg_inc_val is None:
                            avg_inc_val = num_proc / total_num_proc * inc_val
                        else:
                            avg_inc_val += num_proc / total_num_proc * inc_val

                if avg_inc_val is None or key.endswith("num_batches_tracked"):
                    continue
                else:
                    param.add_(avg_inc_val)

        if idx % self.config.EVAL_DISP_INTERVAL == 0:




            if self.valid_data is not None:

                val_samples_processed, val_loss = self.model._evaluate(self.model, self.valid_data)
                list_loss.append(val_loss)



                # report
                print("Epoch: [{}]\tValidation Loss: {}\t".format(idx, val_loss))

            output = self.model.predict(self.test_loader, self.model)

            from octis.evaluation_metrics.diversity_metrics import TopicDiversity
            from octis.evaluation_metrics.coherence_metrics import Coherence
            from octis.evaluation_metrics.classification_metrics import AccuracyScore
            # Define dataset
            dataset = self.data_set

            # Initialize metric
            npmi = Coherence(texts=dataset.get_corpus(), topk=10, measure='c_npmi')
            # Initialize metric
            topic_diversity = TopicDiversity(topk=10)
            # Retrieve metrics score
            topic_diversity_score = topic_diversity.score(output)
            accuracy = AccuracyScore(dataset)
            accuracy_score = accuracy.score(output)

            list_accuracy.append(accuracy_score)



            npmi_score = npmi.score(output)
            list_npmi.append(npmi_score)
            list_topic_diversity.append(topic_diversity_score)






            print("Round #{} (Experiment = {}).".format(idx, self.experiment_name))
            print("Topic diversity: " + str(topic_diversity_score))
            print("Coherence: " + str(npmi_score))
            print("accuracy_score={}".format(accuracy_score))
            print("Estimated time = {}".format(sum(list_est_time)))
            print("Elapsed time = {}".format(timer() - start))
            print("Current lr = {}".format(lr))

        if self.use_adaptive and is_adj_round:
            alg_start = timer()

            for d in list_accumulated_sgrad:
                for k, sg in d.items():
                    self.control.accumulate(k, sg)

            print("Running adaptive pruning algorithm")
            max_dec_diff = self.args.MAX_DEC_DIFF * (0.5 ** (idx / self.config.ADJ_HALF_LIFE))
            self.control.adjust(max_dec_diff, max_density=density_limit)
            print("Total alg time = {}. Max density = {}.".format(timer() - alg_start, density_limit))
            print("Num params:")
            disp_num_params(self.model)

        est_time = self.config.TIME_CONSTANT
        for layer, comp_coeff in zip(self.model.prunable_layers, self.config.COMP_COEFFICIENTS):
            est_time += layer.num_weight * (comp_coeff + self.config.COMM_COEFFICIENT)


        model_size = self.model.calc_num_all_active_params(True)
        list_est_time.append(est_time)
        list_model_size.append(model_size)

        if idx % self.save_interval == 0:
            mkdir_save(list_npmi, os.path.join(self.save_path, "npmi.pt"))
            mkdir_save(list_topic_diversity, os.path.join(self.save_path, "topic_diversity.pt"))

            mkdir_save(list_accuracy, os.path.join(self.save_path, "list_accuracy.pt"))
            mkdir_save(list_recall, os.path.join(self.save_path, "list_recall.pt"))
            mkdir_save(list_precision, os.path.join(self.save_path, "list_precision.pt"))
            mkdir_save(list_f1, os.path.join(self.save_path, "list_f1"))
            mkdir_save(list_loss, os.path.join(self.save_path, "loss.pt"))
            mkdir_save(list_est_time, os.path.join(self.save_path, "est_time.pt"))
            mkdir_save(list_model_size, os.path.join(self.save_path, "model_size.pt"))
            mkdir_save(self.model, os.path.join(self.save_path, "model.pt"))

        return [layer.mask for layer in self.model.prunable_layers], [self.model.state_dict() for _ in
                                                                      range(self.config.NUM_CLIENTS)]


class ProdLDAClient:
    def __init__(self, model, config, use_adaptive, args):
        self.config = config
        self.device = device
        self.use_adaptive = use_adaptive
        self.model = deepcopy(model).to(self.device)
        self.optimizer = None
        self.optimizer_scheduler = None
        self.optimizer_wrapper = None
        self.train_loader = None
        self.args = args
        self.tset_loader = None

        self.list_mask = [None for _ in range(len(self.model.prunable_layers))]
        if self.use_adaptive:
            self.accumulated_sgrad = dict()

    @abstractmethod
    def init_optimizer(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_train_loader(self, *args, **kwargs):
        pass

    def main(self, is_adj_round):

        num_proc_data = 0

        lr = self.args.lr

        accumulated_grad = dict()

        for _ in range(self.config.NUM_LOCAL_UPDATES):
            with torch.no_grad():
                for layer, mask in zip(self.model.prunable_layers, self.list_mask):
                    if mask is not None:
                        layer.weight *= mask

            inputs = self.train_loader.get_next_batch().float()


            self.model._train(inputs, self.model, self.optimizer, self.device)




            num_proc_data += len(inputs)

            for key, param in self.model.named_parameters():
                if param.grad == None:
                    continue

                if key in accumulated_grad.keys():


                    accumulated_grad[key] += param.grad  # g
                else:
                    accumulated_grad[key] = deepcopy(param.grad)  # g
        #将累计梯度加入到累积梯度函数当中
        with torch.no_grad():
            if self.use_adaptive:
                for key, grad in accumulated_grad.items():
                    if key in self.accumulated_sgrad.keys():
                        self.accumulated_sgrad[key] += (grad ** 2) * num_proc_data
                    else:
                        self.accumulated_sgrad[key] = (grad ** 2) * num_proc_data

            for layer, mask in zip(self.model.prunable_layers, self.list_mask):
                if mask is not None:
                    layer.weight *= mask



        if self.use_adaptive and is_adj_round:
            sgrad_to_upload = deepcopy_dict(self.accumulated_sgrad)
            self.accumulated_sgrad = dict()
        else:
            sgrad_to_upload = {}
        return self.model.state_dict(), num_proc_data, sgrad_to_upload, lr

    def load_mask(self, masks):
        self.list_mask = masks

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class ProdLDAFL(ABC):
    def __init__(self, args, config, server, client_list):
        self.config = config

        self.use_adaptive = args.use_adaptive
        self.tgt_d, self.max_d = args.target_density, args.max_density
        self.max_round = config.MAX_ROUND
        self.server = server
        self.client_list = client_list

        self.list_topic_diversity,self.list_npmi,self.list_accuracy, self.list_recall, self.list_precision, self.list_f1, self.list_loss, self.list_est_time, self.list_model_size = [],[],[],[], [], [],[], [], []
        self.start_adj_round = None

    def main(self):
        len_pre_rounds = 0


        start = timer()
        for idx in range(self.max_round):
            list_state_dict, list_num, list_accum_sgrad, list_last_lr = [], [], [], []
            is_adj_round = False
            if idx % self.config.EVAL_DISP_INTERVAL == 0:
                is_adj_round = self.check_adj_round(len_pre_rounds, idx)


            for client in self.client_list:

                sd, npc, grad, last_lr = client.main(is_adj_round)


                list_state_dict.append(sd)
                list_num.append(npc)
                list_accum_sgrad.append(grad)
                list_last_lr.append(last_lr)

            last_lr = list_last_lr[0]
            for client_lr in list_last_lr[1:]:
                assert client_lr == last_lr

            density_limit = None
            if self.max_d is not None:
                density_limit = self.max_d
            if self.tgt_d is not None:
                assert self.tgt_d <= self.max_d
                density_limit += (self.tgt_d - self.max_d) / self.max_round * idx*1.1
                #density_limit += (self.tgt_d - self.max_d) / self.max_round * idx *4
                density_limit = max(density_limit,self.tgt_d)


            list_mask, new_list_sd = self.server.main(idx, list_state_dict, list_num, last_lr, list_accum_sgrad, start,
                                                      self.list_topic_diversity,self.list_npmi,self.list_accuracy, self.list_recall, self.list_precision, self.list_f1, self.list_loss, self.list_est_time, self.list_model_size, is_adj_round, density_limit)
            for client, new_sd in zip(self.client_list, new_list_sd):
                client.load_state_dict(new_sd)
                client.load_mask(list_mask)




    def check_adj_round(self, pre_rounds, idx):
        if not self.use_adaptive or idx == 0:
            return False
        if len(self.list_loss) * self.config.EVAL_DISP_INTERVAL \
                < pre_rounds * self.config.EVAL_DISP_INTERVAL + self.config.ADJ_INTERVAL:
            return False
        elif self.start_adj_round is None:
            self.start_adj_round = idx
            print("Starting reconfiguration at round {}.".format(idx))
            return True
        else:
            return (idx - self.start_adj_round) % self.config.ADJ_INTERVAL == 0
