import os
import torch
from bases.fl.simulation_real.adaptive import AdaptiveServer, AdaptiveClient, AdaptiveFL, parse_args
from bases.optim.optimizer import SGD
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from control.algorithm import ControlModule
from bases.nn.models import Conv4
from bases.vision.FLsampler import FLSampler
from configs.celeba import *
import configs.celeba as config

from utils.save_load import mkdir_save


class CelebAAdaptiveServer(AdaptiveServer):
    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="test", num_workers=0, batch_size=100, shuffle=False,
                                           pin_memory=True)

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
        self.ip_train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE,
                                               shuffle=True, num_workers=0, user_list=list(range(10)), pin_memory=True)
        self.ip_test_loader = get_data_loader(EXP_NAME, data_type="test", num_workers=0, batch_size=100,
                                              pin_memory=True)
        ip_optimizer = SGD(self.model.parameters(), lr=INIT_LR, weight_decay=1e-5)
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


class CelebAAdaptiveClient(AdaptiveClient):
    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR, weight_decay=0)
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)

    def init_train_loader(self, tl):
        self.train_loader = tl


def get_indices_list():
    train_meta = torch.load(os.path.join("datasets", "CelebA", "processed", "train_meta.pt"))
    cur_pointer = 0
    indices_list = []
    for ul in list_users:
        num_data = 0
        for user_id in ul:
            num_data += len(train_meta[user_id]["x"])
        indices_list.append(list(range(cur_pointer, cur_pointer + num_data)))
        cur_pointer += num_data

    return indices_list




if __name__ == "__main__":
    if os.getcwd().startswith('/mnt/sda1/mcj/PruneFL-master/PruneFL-master'):
        os.chdir('/mnt/sda1/mcj/PruneFL-master/PruneFL-master')
    args = parse_args()
    torch.manual_seed(args.seed)
    
    num_user_path = os.path.join("datasets", "CelebA", "processed", "num_users.pt")
    if not os.path.isfile(num_user_path):
        get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False, num_workers=0,
                        pin_memory=True)
    num_users = torch.load(num_user_path)

    server = CelebAAdaptiveServer(args, config, Conv4())
    list_models, list_users = server.init_clients()

    sampler = FLSampler(get_indices_list(), MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                        NUM_CLIENTS)
    print("Sampler initialized")

    train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                   sampler=sampler, num_workers=8, pin_memory=True)

    client_list = [CelebAAdaptiveClient(list_models[idx], config, args.use_adaptive,args) for idx in range(NUM_CLIENTS)]
    for client in client_list:
        client.init_optimizer()
        client.init_train_loader(train_loader)

    print("All initialized. Experiment is {}. Use adaptive = {}. Use initial pruning = {}. Client selection = {}. "
          "Num users = {}. Seed = {}. Max round = {}. "
          "Target density = {}".format(EXP_NAME, args.use_adaptive, args.initial_pruning, args.client_selection,
                                       num_users, args.seed, MAX_ROUND, args.target_density))

    fl_runner = AdaptiveFL(args, config, server, client_list)
    fl_runner.main()
