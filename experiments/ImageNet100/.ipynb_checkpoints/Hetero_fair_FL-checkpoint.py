import os
import torch
from bases.fl.simulation_real.Prune_Fair_FL_cuda_fast import FedMapServer, FedMapClient,FedMapFL, parse_args
from bases.optim.optimizer import SGD
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from bases.vision.sampler import FLSampler
from control.sub_algorithm import ControlModule
from bases.nn.models.resnet import resnet18
from configs.imagenet100 import *
import configs.imagenet100 as config
from torch.optim import lr_scheduler
from control.utils import ControlScheduler

from utils.save_load import mkdir_save


class INFedMapServer(FedMapServer):
    def get_init_extra_params(self):

        return [([i for i in range(19 * j, 19 * (j + 1) if j != 9 else 193)], self.client_is_sparse) for j in range(10)]

    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="val",batch_size=200, num_workers=0, pin_memory=True)

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
        self.control = ControlModule(self.model, config=config)

    def init_ip_config(self):
        self.ip_train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE,
                                               subset_indices=self.indices[0][:IP_DATA_BATCH * CLIENT_BATCH_SIZE],
                                               shuffle=True, num_workers=0, pin_memory=True)

        self.ip_test_loader = get_data_loader(EXP_NAME, data_type="val", batch_size=200, num_workers=0,
                                              pin_memory=True)

        ip_optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        self.ip_optimizer_wrapper = OptimizerWrapper(self.model, ip_optimizer)
        self.ip_control = ControlModule(model=self.model, config=config)

    def save_exp_config(self):
        exp_config = {"exp_name": EXP_NAME, "seed": args.seed, "batch_size": CLIENT_BATCH_SIZE,
                      "num_local_updates": NUM_LOCAL_UPDATES, "mdd": MAX_DEC_DIFF, "init_lr": INIT_LR,
                      "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY, "lrhl": LR_HALF_LIFE, "step_size": STEP_SIZE,
                      "ahl": ADJ_HALF_LIFE, "use_adaptive": self.use_adaptive,
                      "client_selection": args.client_selection}
        if args.client_selection:
            exp_config["num_users"] = num_users
        mkdir_save(exp_config, os.path.join(self.save_path, "exp_config.pt"))



class INFedMapClient(FedMapClient):
    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        import torch.optim.lr_scheduler as lr_scheduler
        self.optimizer_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=STEP_SIZE,
                                                       gamma=0.5 ** (STEP_SIZE / LR_HALF_LIFE))
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)

        if self.args.lr_scheduler:
            import torch.optim.lr_scheduler as lr_scheduler

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





class args:
    def __init__(self, parse_args=None):
        self.seed = 0
        self.experiment_name = 'Hetero_FL——[Hetero_fair_cuda20000_second]'
        self.min_density = 1
        self.density = [1,1]
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
        self.interval = 100
        self.device = 2
        self.weight_decay = -1
        self.increase = 'un_increase'
        self.accumulate = 'g'

        self.weight_decay = -1

        self.merge = 'sub_fed_avg'
        self.control_lr =0
        self.fine_tune = 0
        self.uc = None
        self.ft = 'n'
        self.lr_scheduler = False
        self.experiment_name = self.experiment_name+'_'+str(self.density) + str(self.fair) + '_' + str(self.fair_degree) + '_' + str(
            self.interval) + '_' + str(self.increase) + '_' + str(self.weight_decay)+'_'+self.merge+'_'+self.accumulate+'_'+str(self.control_lr)+'_'+str(self.fine_tune)+'_'+self.ft+'_'+str(self.lr_scheduler)

if __name__ == "__main__":
    if os.getcwd().startswith('/mnt/sda1/mcj/PruneFL-master/PruneFL-master'):
        os.chdir('/mnt/sda1/mcj/PruneFL-master/PruneFL-master')

    if os.getcwd().startswith("/data/mcj/Prune_fl"):
        os.chdir("/data/mcj/Prune_fl")
    if os.getcwd().startswith("F:\\xhx\\Prune_FL"):
        os.chdir("'F:\\xhx\\Prune_FL'")

    args = args()
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
    seed, resume, use_adaptive = 0, False, True
    torch.manual_seed(args.seed)
    num_users = 200
    num_slices = num_users if args.client_selection else NUM_CLIENTS

    server = INFedMapServer(config, args, resnet18(num_classes=200), seed, SGD, {"lr": config.INIT_LR}, use_adaptive, device)
    list_models, list_indices = server.init_clients()


    sampler = FLSampler(list_indices, MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                        num_slices)
    print("Sampler initialized")

    train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                   sampler=sampler, num_workers=0, pin_memory=True)

    client_list = [INFedMapClient(list_models[idx], config, args.use_adaptive, server.list_extra_params[idx], exp_config = server.exp_config, args=args, device = device) for idx in range(NUM_CLIENTS)]

    for client in client_list:
        client.init_optimizer()
        client.init_train_loader(train_loader)
        client.init_test_loader(server.test_loader)
    if args.weight_decay >= 0:
        for client in client_list[-3:]:
            for param_group in client.optimizer_wrapper.optimizer.param_groups:
                param_group['weight_decay'] = args.weight_decay

            for param_group in client.optimizer.param_groups:
                print("Learning Rate:", param_group['lr'])
                print("Weight Decay:", param_group['weight_decay'])
                print()




    fl_runner = FedMapFL(args, config, server, client_list)
    fl_runner.main()
