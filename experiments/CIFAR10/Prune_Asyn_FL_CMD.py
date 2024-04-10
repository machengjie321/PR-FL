import os
import torch
from bases.fl.simulation_real.Prune_Asyn_FL import FedMapServer, FedMapClient,FedMapFL, parse_args
from bases.optim.optimizer import SGD
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from bases.vision.sampler import FLSampler
from control.sub_algorithm import ControlModule


from bases.nn.models.vgg import VGG11
from configs.cifar10 import *
import configs.cifar10 as config

from utils.save_load import mkdir_save


class INFedMapServer(FedMapServer):
    def get_init_extra_params(self):

        return [([i for i in range(19 * j, 19 * (j + 1) if j != 9 else 193)], self.client_is_sparse) for j in range(10)]

    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="test",batch_size=1000, num_workers=config.test_num, pin_memory=True)

    def init_clients(self):
        rand_perm = torch.randperm(NUM_TRAIN_DATA).tolist()
        indices = []
        len_slice = NUM_TRAIN_DATA // num_slices

        for i in range(num_slices):
            indices.append(rand_perm[i * len_slice: (i + 1) * len_slice])

        models = [self.model for _ in range(NUM_CLIENTS)]
        self.indices = indices
        return models, indices

    def init_control(self):
        self.control = ControlModule(self.model, config=config, args=args)

    def init_ip_config(self):
        self.ip_train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE,
                                               subset_indices=self.indices[0][:IP_DATA_BATCH * CLIENT_BATCH_SIZE],
                                               shuffle=True, num_workers=config.train_num, pin_memory=True)

        self.ip_test_loader = get_data_loader(EXP_NAME, data_type="test", batch_size=1000, num_workers=config.test_num,
                                              pin_memory=True)

        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        import torch.optim.lr_scheduler as lr_scheduler

        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)
        self.ip_control = ControlModule(model=self.model, config=config, args=args)

    def save_exp_config(self):
        exp_config = {"exp_name": EXP_NAME, "seed": args.seed, "batch_size": CLIENT_BATCH_SIZE,
                      "num_local_updates": NUM_LOCAL_UPDATES, "mdd": MAX_DEC_DIFF, "init_lr": INIT_LR,
                      "lrhl": LR_HALF_LIFE, "ahl": ADJ_HALF_LIFE, "use_adaptive": self.use_adaptive,
                      "client_selection": args.client_selection}
        if args.client_selection:
            exp_config["num_users"] = num_users
        mkdir_save(exp_config, os.path.join(self.save_path, "exp_config.pt"))



class INFedMapClient(FedMapClient):
    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        if self.args.lr_scheduler:
            import torch.optim.lr_scheduler as lr_scheduler
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=100
            ,verbose=True)
        else:
            self.scheduler = None

        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, self.scheduler,)





    def init_train_loader(self, tl):
        self.train_loader = tl

    def init_test_loader(self, tl):
        self.test_loader = tl




class args:
    def __init__(self, parse_args):
        self.seed = 0
        self.experiment_name = 'PIF_CIFAR10'
        self.min_density = parse_args.min_density


        self.use_adaptive = True
        self.client_selection = False
        self.initial_pruning = False
        self.target_density = 0.5
        self.max_density = 1
        self.client_density = config.client_density
        self.density = [x for i, x in enumerate(self.client_density) if x not in self.client_density[:i]]
        # n:number,u:update,un:update_number
        self.fair = parse_args.use_fair
        self.fair_degree = parse_args.fair_degree
        self.interval = parse_args.interval

        self.device = parse_args.device
        self.increase = parse_args.increase
        self.accumulate = parse_args.accumulate

        self.weight_decay = parse_args.weight_decay

        self.merge = parse_args.merge
        self.control_lr = parse_args.control_lr
        self.wdn = parse_args.wdn
        self.ft = parse_args.ft
        self.lr_scheduler = parse_args.lr_scheduler
        self.uc = parse_args.uc
        self.resume = parse_args.resume
        self.lr_warm = parse_args.lr_warm
        self.esc = parse_args.esc
        self.chronous = parse_args.chronous
        self.recover = parse_args.recover
        self.stal = parse_args.stal
        self.stal_a = parse_args.stal_a
        self.holistic_coeff = 10
        self.list_client_coeff = config.list_client_coeff

        assert self.stal.lower().startswith('con') or self.stal.lower().startswith('poly') or self.stal.lower().startswith('hinge')
        assert self.stal_a <= 1

        self.experiment_name = self.experiment_name + '_' + str(self.list_client_coeff)+str(self.holistic_coeff)+'_'+('' if self.chronous == 'syn' else '_'+self.chronous+'_') +str('_recover' if self.recover else '')\
            +'_'+str(self.stal.lower())+ '_' + self.merge  +'_'+ str('' if self.stal_a == 0.6 else self.stal_a)+str('' if self.fair == 'un_fair' else '_'+self.fair) + '_' + str(
            '' if self.fair_degree == 0 else self.fair_degree) + '_' + str(
            self.interval) + '_' + str('' if self.increase == 0.2 else self.increase) + '_' + str('' if self.weight_decay == 0 else
            self.weight_decay) + '_' + str('' if self.accumulate=='g' else self.accumulate ) + str('_clr' if self.control_lr else '') + '_' + str('' if
            self.wdn == 10 else self.wdn) + str('_ft' if self.ft == 'n' else '') + str('_lr_scheduler' if self.lr_scheduler else '') +  str(
            '_lr_warm' if self.lr_warm else '') + str('_uc' if self.uc == 'n' else '')+ str('_esc' if self.esc else '')

if __name__ == "__main__":
    if os.getcwd().startswith('/mnt/sda1/mcj/PruneFL-master/PruneFL-master'):
        os.chdir('/mnt/sda1/mcj/PruneFL-master/PruneFL-master')

    if os.getcwd().startswith("/data/mcj/Prune_fl"):
        os.chdir("/data/mcj/Prune_fl")
    if os.getcwd().startswith("F:\\xhx\\Prune_FL"):
        os.chdir("'F:\\xhx\\Prune_FL'")


    args = args(parse_args())
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
    seed, resume, use_adaptive = 0, False, True
    torch.manual_seed(args.seed)
    num_users = 100
    num_slices = num_users if args.client_selection else NUM_CLIENTS

    server = INFedMapServer(config, args, VGG11(), seed, SGD, {"lr": config.INIT_LR}, use_adaptive, device)
    list_models, list_indices = server.init_clients()

    list_train_loader = []
    for i in range(len(list_indices)):
        sampler = FLSampler(list_indices, i, MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                            num_slices)

        train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                       sampler=sampler, num_workers=config.train_num, pin_memory=True)
        list_train_loader.append(train_loader)



    client_list = [INFedMapClient(list_models[idx], config, args.use_adaptive, server.list_extra_params[idx], exp_config=server.exp_config, args=args, device=device) for idx in range(NUM_CLIENTS)]

    i = 0
    for client in client_list:
        client.init_optimizer()
        client.init_train_loader(list_train_loader[i])
        client.init_test_loader(server.test_loader)
        i = i+1

    if args.weight_decay >= 0:
        for client in client_list[-args.wdn:]:
            for param_group in client.optimizer_wrapper.optimizer.param_groups:
                param_group['weight_decay'] = args.weight_decay

            for param_group in client.optimizer.param_groups:
                print("Learning Rate:", param_group['lr'])
                print("Weight Decay:", param_group['weight_decay'])
                print()




    fl_runner = FedMapFL(args, config, server, client_list)
    fl_runner.main()
