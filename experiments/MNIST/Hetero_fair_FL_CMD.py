import os
import torch
from bases.fl.simulation_real.Prune_Fair_FL_cuda_fast import FedMapServer, FedMapClient,FedMapFL, parse_args
from bases.optim.optimizer import SGD
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from bases.nn.models.leaf import Conv2
from bases.vision.sampler import FLSampler
from control.sub_algorithm import ControlModule
from configs.femnist import *
import configs.femnist as config
from control.utils import ControlScheduler
import torch.optim.lr_scheduler as lr_scheduler

from utils.save_load import mkdir_save


class FEMNISTFedMapServer(FedMapServer):
    def get_init_extra_params(self):

        return [([i for i in range(19 * j, 19 * (j + 1) if j != 9 else 193)], self.client_is_sparse) for j in range(10)]

    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="test", num_workers=test_num, pin_memory=True)

    def init_clients(self):

        if self.client_selection:
            list_usr = [[i] for i in range(num_users)]
        else:
            nusr = num_users // NUM_CLIENTS  # num users for the first NUM_CLIENTS - 1 clients
            list_usr = [list(range(nusr * j, nusr * (j + 1) if j != NUM_CLIENTS - 1 else num_users)) for j in
                        range(NUM_CLIENTS)]
        models = [self.model for _ in range(NUM_CLIENTS)]
        return models, list_usr

    def init_control(self):
        self.control = ControlModule(self.model, config=config)

    def init_ip_config(self):
        self.ip_train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=True,
                                               num_workers=train_num, user_list=[0], pin_memory=True)
        self.ip_test_loader = get_data_loader(EXP_NAME, data_type="test", num_workers=test_num, pin_memory=True)
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
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)

        if self.args.lr_scheduler:


            # 定义学习率调度器
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=10,
                                                            verbose=True)
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
    def __init__(self, parse_args):
        self.seed = 0
        self.experiment_name = 'HFF_FEMNIST'
        self.min_density = parse_args.min_density
        self.density = [1, 0.3]
        self.density.append(self.min_density)
        self.use_adaptive = True
        self.client_selection = False
        self.initial_pruning = False
        self.target_density = 0.5
        self.max_density = 1
        self.client_density = [self.density[0]]*2+[self.density[1]]*4+[self.density[2]]*4
        # n:number,u:update,un:update_number
        self.fair = parse_args.use_fair
        self.fair_degree = parse_args.fair_degree
        self.interval = parse_args.interval*2
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
        if self.uc == 'n':
            self.experiment_name = self.experiment_name + '_' + str(self.density) + str(self.fair) + '_' + str(
                self.fair_degree) + '_' + str(
                self.interval) + '_' + str(self.increase) + '_' + str(
                self.weight_decay) + '_' + self.merge + '_' + self.accumulate + '_' + str(self.control_lr) + '_' + str(
                self.wdn) + '_' + self.ft + '_' + str(self.lr_scheduler) + '_uc' + '_' + str(self.lr_warm)
        else:
            self.experiment_name = self.experiment_name + '_' + str(self.density) + str(self.fair) + '_' + str(
                self.fair_degree) + '_' + str(
                self.interval) + '_' + str(self.increase) + '_' + str(
                self.weight_decay) + '_' + self.merge + '_' + self.accumulate + '_' + str(self.control_lr) + '_' + str(
                self.wdn) + '_' + self.ft + '_' + str(self.lr_scheduler) + '_' + str(self.lr_warm)




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


    num_user_path = os.path.join("datasets", "FEMNIST", "processed", "num_users.pt")

    if not os.path.isfile(num_user_path):
        get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False, num_workers=train_num,
                        pin_memory=True)
    num_users = torch.load(num_user_path)




    server = FEMNISTFedMapServer(config, args, Conv2(), seed, SGD, {"lr": config.INIT_LR}, use_adaptive, device)

    list_models, list_users = server.init_clients()

    sampler = FLSampler(get_indices_list(), MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                        NUM_CLIENTS)
    print("Sampler initialized")

    train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                   sampler=sampler, num_workers=train_num, pin_memory=True)

    client_list = [FEMNISTFedMapClient(list_models[idx], config, args.use_adaptive, server.list_extra_params[idx], exp_config = server.exp_config, args=args, device = device) for idx in range(NUM_CLIENTS)]

    for client in client_list:
        client.init_optimizer()
        client.init_train_loader(train_loader)
        client.init_test_loader(server.test_loader)

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
