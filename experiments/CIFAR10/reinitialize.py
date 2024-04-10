import os
import torch
from bases.fl.simulation.reinitialize import ReinitServer, ReinitClient, ReinitFL, parse_args
from bases.optim.optimizer import SGD
from torch.optim import lr_scheduler
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from bases.vision.sampler import FLSampler
from bases.nn.models.vgg import VGG11
from configs.cifar10 import *
import configs.cifar10 as config

from utils.save_load import load


class CIFAR10ReinitServer(ReinitServer):
    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="test", batch_size=1000, num_workers=8, pin_memory=True)

    def init_clients(self):
        rand_perm = torch.randperm(NUM_TRAIN_DATA).tolist()
        indices = []
        len_slice = NUM_TRAIN_DATA // num_slices

        for i in range(num_slices):
            indices.append(rand_perm[i * len_slice: (i + 1) * len_slice])

        models = [self.model for _ in range(NUM_CLIENTS)]
        return models, indices


class CIFAR10ReinitClient(ReinitClient):
    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        self.optimizer_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5 ** (1 / LR_HALF_LIFE))
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, self.optimizer_scheduler)

    def init_train_loader(self, tl):
        self.train_loader = tl


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "rr":
        prev_config = load(os.path.join("results", EXP_NAME,
                                        "adaptive_cs" if args.client_selection else "adaptive", "exp_config.pt"))
        args.seed = prev_config["seed"] + 1
    torch.manual_seed(args.seed)

    num_users = 100
    num_slices = num_users if args.client_selection else NUM_CLIENTS

    server = CIFAR10ReinitServer(args, config, VGG11())
    list_models, list_indices = server.init_clients()

    sampler = FLSampler(list_indices, MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                        num_slices)
    print("Sampler initialized")

    train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                   sampler=sampler, num_workers=8, pin_memory=True)

    client_list = [CIFAR10ReinitClient(config, list_models[idx],args) for idx in range(NUM_CLIENTS)]
    for client in client_list:
        client.init_optimizer()
        client.init_train_loader(train_loader)

    print("All initialized. Mode = {}. Client selection = {}. Num users = {} .Seed = {}. "
          "Max round = {}.".format(EXP_NAME, "Reinit" if args.mode == "r" else "Random reinit", args.client_selection,
                                   args.seed, MAX_ROUND))

    fl_runner = ReinitFL(config, server, client_list)
    fl_runner.main()
