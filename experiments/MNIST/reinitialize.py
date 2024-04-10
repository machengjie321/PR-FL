import os
import torch
from bases.fl.simulation.reinitialize import ReinitServer, ReinitClient, ReinitFL, parse_args
from bases.optim.optimizer import SGD
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from bases.vision.FLsampler import FLSampler
from bases.nn.models.leaf import Conv2
from configs.femnist import *
import configs.femnist as config

from utils.save_load import load


class FEMNISTReinitServer(ReinitServer):
    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="test", num_workers=8, pin_memory=True)

    def init_clients(self):
        if args.client_selection:
            list_usr = [[i] for i in range(num_users)]
        else:
            nusr = num_users // NUM_CLIENTS  # num users for the first NUM_CLIENTS - 1 clients
            list_usr = [list(range(nusr * j, nusr * (j + 1) if j != NUM_CLIENTS - 1 else num_users)) for j in
                        range(NUM_CLIENTS)]
        models = [self.model for _ in range(NUM_CLIENTS)]
        return models, list_usr


class FEMNISTReinitClient(ReinitClient):
    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)

    def init_train_loader(self, tl):
        self.train_loader = tl


def get_indices_list():
    cur_pointer = 0
    indices_list = []
    for ul in list_users:
        num_data = 0
        for user_id in ul:
            train_meta = torch.load(os.path.join("datasets", "FEMNIST", "processed", "train_{}.pt".format(user_id)))
            num_data += len(train_meta[0])
        indices_list.append(list(range(cur_pointer, cur_pointer + num_data)))
        cur_pointer += num_data

    return indices_list


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "rr":
        prev_config = load(os.path.join("results", EXP_NAME,
                                        "adaptive_cs" if args.client_selection else "adaptive", "exp_config.pt"))
        args.seed = prev_config["seed"] + 1
    torch.manual_seed(args.seed)
    #modify the code at 8:00 pm on February 27, Because I think it is no use
    # num_user_path = os.path.join("datasets", "FEMNIST", "processed", "num_users.pt")
    # if not os.path.isfile(num_user_path):
    #     get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False, num_workers=8,
    #                     pin_memory=True)
    # num_users = torch.load(num_user_path)
    num_users = 10

    server = FEMNISTReinitServer(args, config, Conv2())
    list_models, list_users = server.init_clients()

    sampler = FLSampler(get_indices_list(), MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                        NUM_CLIENTS)
    print("Sampler initialized")

    train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                   sampler=sampler, num_workers=8, pin_memory=True)

    client_list = [FEMNISTReinitClient(config, list_models[idx], args) for idx in range(NUM_CLIENTS)]
    for client in client_list:
        client.init_optimizer()
        client.init_train_loader(train_loader)

    print("All initialized. Mode = {}. Client selection = {}. Num users = {} .Seed = {}. "
          "Max round = {}.".format(EXP_NAME, "Reinit" if args.mode == "r" else "Random reinit", args.client_selection,
                                   args.seed, MAX_ROUND))

    if args.mode == "rr":
        prev_config = load(os.path.join("results", EXP_NAME, server.adaptive_folder, "exp_config.pt"))
        args.seed = prev_config["seed"] + 1

    fl_runner = ReinitFL(config, server, client_list)
    fl_runner.main()