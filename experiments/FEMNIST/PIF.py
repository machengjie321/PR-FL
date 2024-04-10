import os
import torch
from bases.fl.simulation_real.Prune_Recover_FL import FedMapServer, FedMapClient,FedMapFL, parse_args
from bases.optim.optimizer import SGD
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from bases.nn.models.leaf import Conv2
from bases.vision.sampler import FLSampler
from control.sub_algorithm import ControlModule
from configs.femnist import *
import configs.femnist as config
from control.utils import ControlScheduler

from utils.save_load import mkdir_save


class FEMNISTFedMapServer(FedMapServer):
    def get_init_extra_params(self):

        return [([i for i in range(19 * j, 19 * (j + 1) if j != 9 else 193)], self.client_is_sparse) for j in range(10)]

    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="test", num_workers=test_num, pin_memory=True)

    def init_clients(self):
        rand_perm = torch.randperm(NUM_TRAIN_DATA).tolist()
        indices = []
        len_slice = NUM_TRAIN_DATA // num_slices

        for i in range(num_slices):
            indices.append(rand_perm[i * len_slice: (i + 1) * len_slice])

        models = [self.model for _ in range(NUM_CLIENTS)]
        self.indices = indices
        if self.client_selection:
            list_usr = [[i] for i in range(num_users)]
        else:
            nusr = num_users // NUM_CLIENTS  # num users for the first NUM_CLIENTS - 1 clients
            list_usr = [list(range(nusr * j, nusr * (j + 1) if j != NUM_CLIENTS - 1 else num_users)) for j in
                        range(NUM_CLIENTS)]
        return models, indices,list_usr

    def init_control(self):
        self.control = ControlModule(self.model, config=config)

    def init_ip_config(self):
        self.ip_train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=True,
                                               num_workers=train_num, user_list=[0], pin_memory=True)
        self.ip_test_loader = get_data_loader(EXP_NAME, data_type="test", num_workers=0, pin_memory=True)
        ip_optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        self.ip_optimizer_wrapper = OptimizerWrapper(self.model, ip_optimizer)
        self.ip_control = ControlModule(model=self.model, config=config)

    def save_exp_config(self):
        exp_config = {"exp_name": EXP_NAME, "seed": args.seed, "batch_size": CLIENT_BATCH_SIZE,
                      "num_local_updates": NUM_LOCAL_UPDATES, "mdd": MAX_DEC_DIFF, "init_lr": INIT_LR,
                      "ahl": ADJ_HALF_LIFE, "use_adaptive": self.use_adaptive,
                      "client_selection": args.client_selection}
        if self.client_selection:
            exp_config["num_users"] = num_users
        mkdir_save(exp_config, os.path.join(self.save_path, "exp_config.pt"))


class FEMNISTFedMapClient(FedMapClient):
    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        if self.args.lr_scheduler:
            import torch.optim.lr_scheduler as lr_scheduler
            self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer,lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=50,
                                                            verbose=True))
        else:
            self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer,)


            # mode：'min'表示监测指标应减小，'max'表示应增大
            # factor：学习率将乘以的因子
            # patience：如果验证损失在patience个epoch内都没有改善，则减小学习率
            # verbose：是否打印学习率更新信息


    def init_train_loader(self, tl):
        self.train_loader = tl

    def init_test_loader(self, tl):
        self.test_loader = tl

def get_indices_list():
    cur_pointer = 0
    indices_list = []
    for ul in list_users:
        num_data = 0
        for user_id in ul:
            train_meta = torch.load(
                os.path.join("datasets", "FEMNIST", "processed", "train_{}.pt".format(user_id)))
            num_data += len(train_meta[0])
        indices_list.append(list(range(cur_pointer, cur_pointer + num_data)))
        cur_pointer += num_data

    return indices_list


class args:
    def __init__(self):
        self.seed = 0
        self.experiment_name = 'Hetero_FL——[Hetero_fair_cuda20000_second]'
        self.min_density = 0.1
        self.density = [1,0.3]
        self.density.append(self.min_density)
        self.use_adaptive = True
        self.client_selection = False
        self.initial_pruning = False
        self.target_density = 0.5
        self.max_density = 1
        self.client_density = [self.density[0]]*3+[self.density[1]]*4+[self.density[2]]*3
        # n:number,u:update,un:update_number
        self.fair = 'un_fair'
        self.fair_degree = 1
        self.interval = 30
        self.device = 2
        self.weight_decay = -1
        self.increase = 0.1
        self.accumulate = 'g'

        self.weight_decay = -1

        self.merge = 'buff_mask_fed_avg'
        self.control_lr =0
        self.wdn = 0
        self.fine_tune = 0
        self.ft = 'n'
        self.lr_scheduler = False
        self.resume = False
        self.uc = 'y'
        self.lr_warm = False
        self.esc = False
        self.chronous = 'asyn'
        self.recover = True
        self.stal = 'poly'
        self.stal_a = 0.6
        assert self.stal.lower().startswith('con') or self.stal.lower().startswith('poly') or self.stal.lower().startswith('hinge')
        assert self.stal_a <= 1
        self.sample = 'niid'
        self.Res = False
        self.experiment_name = self.experiment_name + '_' + str(self.density)+str(self.min_density)+'_'+('' if self.chronous == 'syn' else '_'+self.chronous+'_') +str('_recover' if self.recover else '')\
            +'_'+str(self.stal.lower())+ '_' + self.merge+ '_' + self.sample +'_'+ str('' if self.stal_a == 0.6 else self.stal_a)+str('' if self.fair == 'un_fair' else '_'+self.fair) + '_' + str(
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
    args = args()
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
    num_users = 200
    num_slices = num_users if args.client_selection else NUM_CLIENTS
    seed, resume, use_adaptive = 0, False, True
    torch.manual_seed(args.seed)


    num_user_path = os.path.join("datasets", "FEMNIST", "processed", "num_users.pt")

    if not os.path.isfile(num_user_path):
        get_data_loader(EXP_NAME, data_typze="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False, num_workers=train_num,
                        pin_memory=True)
    num_users = torch.load(num_user_path)




    server = FEMNISTFedMapServer(config, args, Conv2(), seed, SGD, {"lr": config.INIT_LR}, use_adaptive, device)
    list_train_loader = []
    list_models, list_indices, list_users = server.init_clients()

    assert args.sample == 'iid' or args.sample == 'niid'
    if args.sample == 'iid':
        pass
    elif args.sample == 'niid':
        list_indices = get_indices_list()

    for i in range(len(list_indices)):
        sampler = FLSampler(list_indices, i, MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                            num_slices)
        train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                       sampler=sampler, num_workers=config.train_num, pin_memory=True)
        list_train_loader.append(train_loader)
    print("Sampler initialized")

    client_list = [FEMNISTFedMapClient(list_models[idx], config, args.use_adaptive, server.list_extra_params[idx], exp_config = server.exp_config, args=args, device = device) for idx in range(NUM_CLIENTS)]

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
