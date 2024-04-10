import os
import torch
from bases.fl.simulation.ProdLDA_Prune import ProdLDAServer, ProdLDAClient, ProdLDAFL, parse_args

from bases.vision.sampler import FLSampler
from control.algorithm import ControlModule
from configs.News_20 import *
import configs.News_20 as config

from utils.save_load import mkdir_save
import bases.nn.models.data as data
from  bases.nn.models.prodLDA import ProdLDA



class NewsProdLDAServer(ProdLDAServer):
    def init_test_loader(self):
        self.test_loader = corpus.test



    def init_clients(self):
        list_usr = [[i] for i in range(config.NUM_CLIENTS)]


        models = [self.model for _ in range(config.NUM_CLIENTS)]
        return models, list_usr

    def init_control(self):
        self.control = ControlModule(self.model, config=config)



    def save_exp_config(self):
        exp_config = {"exp_name": EXP_NAME, "seed": args.seed, "batch_size": CLIENT_BATCH_SIZE,
                      "num_local_updates": NUM_LOCAL_UPDATES, "mdd": MAX_DEC_DIFF, "init_lr": INIT_LR,
                      "ahl": ADJ_HALF_LIFE, "use_adaptive": self.use_adaptive,
                      "client_selection": args.client_selection}


        mkdir_save(exp_config, os.path.join(self.save_path, "exp_config.pt"))


class NewsProdLDAClient(ProdLDAClient):
    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)


    def init_train_loader(self, tl):
        self.train_loader = tl

    def init_test_loader(self):
        self.test_loader = corpus.test


def get_indices_list():
    cur_pointer = 0
    indices_list = []
    train_length = len(train_data)
    client_number = len(list_users)
    avg_len = train_length // client_number
    for ul in list_users:
        if ul == list_users[-1]:
            num_data = train_length -avg_len*(client_number-1)
        else:
            num_data = avg_len

        indices_list.append(list(range(cur_pointer, cur_pointer + num_data)))
        cur_pointer += num_data

    return indices_list


class args:
    def __init__(self):
        self.seed = 0
        self.experiment_name = 'prodLDA_PRUNE30'
        self.use_adaptive = False
        self.client_selection = False
        self.data = '/mnt/sda1/mcj/PruneFL-master/PruneFL-master/bases/nn/models/data/20news'
        self.hidden_size = 256
        self.num_topics = 20
        self.dropout = 0.2
        self.use_lognormal = True
        self.lr = 1e-4
        self.wd = 0
        self.EXP_NAME = 'prodLDA'
        self.target_density = None
        self.max_density = None
        self.epoch_size = 1




if __name__ == "__main__":
    args = args()
    torch.manual_seed(args.seed)

    corpus = data.Corpus(args.data)
    vocab_size = len(corpus.vocab)

    model = ProdLDA(
        None,vocab_size, args.hidden_size, args.num_topics,
        args.dropout, args.use_lognormal)
    server = NewsProdLDAServer(args, config, model)
    list_models, list_users = server.init_clients()
    import numpy as np
    train_data = corpus.train.data
    shuffle_ix = np.random.permutation(np.arange(len(train_data)))
    train_data = train_data[shuffle_ix]


    sampler = FLSampler(get_indices_list(), MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                        NUM_CLIENTS)


    print("Sampler initialized")

    from bases.vision.data_loader import DataLoader
    train_loader = DataLoader(train_data,batch_size=CLIENT_BATCH_SIZE,shuffle=False,sampler=sampler,num_workers=0, pin_memory=True)

    client_list = [NewsProdLDAClient(list_models[idx], config, args.use_adaptive,args) for idx in range(NUM_CLIENTS)]
    for client in client_list:
        client.init_optimizer()
        client.init_train_loader(train_loader)
        client.init_test_loader()


    print("All initialized. Experiment is {}. Use adaptive = {}.  Client selection = {}. "
          "Num users = {}. Seed = {}. Max round = {}. "
          "MAX_DEC= {}. Batch_size = {}. Num_local_updates = {}".format(EXP_NAME, args.use_adaptive, args.client_selection,
                                       config.NUM_CLIENTS, args.seed, MAX_ROUND, config.MAX_DEC_DIFF, config.CLIENT_BATCH_SIZE,config.NUM_LOCAL_UPDATES))


    fl_runner = ProdLDAFL(args, config, server, client_list,corpus.test)
    fl_runner.main()
