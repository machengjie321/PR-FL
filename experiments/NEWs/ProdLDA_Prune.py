import os
import torch
from bases.fl.simulation.ProdLDA_Prune import ProdLDAServer, ProdLDAClient, ProdLDAFL, parse_args

from bases.vision.FLsampler import FLSampler
from control.algorithm import ControlModule
from configs.News_20 import *
import configs.News_20 as config

from utils.save_load import mkdir_save
import bases.nn.models.data as data
from  bases.nn.models.Prod_LDA import ProdLDA
from sklearn.feature_extraction.text import CountVectorizer

from octis.models.model import AbstractModel
from octis.models.pytorchavitm import datasets


class NewsProdLDAServer(ProdLDAServer):
    def init_test_loader(self,test):
        self.test_loader = test


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

    def init_test_loader(self,test):
        self.test_loader = test


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




class tool:
    def __init__(self):
        self.seed = 0

        self.use_adaptive = False
        self.client_selection = False

        self.wd = 0
        self.EXP_NAME = 'prodLDA'
        self.target_density = None
        self.max_density = None


    def preprocess(self, vocab, train, test=None, validation=None):
        vocab2id = {w: i for i, w in enumerate(vocab)}
        vec = CountVectorizer(vocabulary=vocab2id, token_pattern=r'(?u)\b\w+\b')
        entire_dataset = train.copy()
        if test is not None:
            entire_dataset.extend(test)
        if validation is not None:
            entire_dataset.extend(validation)

        vec.fit(entire_dataset)
        idx2token = {v: k for (k, v) in vec.vocabulary_.items()}
        X_train = vec.transform(train)
        train_data = datasets.BOWDataset(X_train.toarray(), idx2token)
        input_size = len(idx2token.keys())

        if test is not None and validation is not None:
            x_test = vec.transform(test)
            test_data = datasets.BOWDataset(x_test.toarray(), idx2token)
            x_valid = vec.transform(validation)
            valid_data = datasets.BOWDataset(x_valid.toarray(), idx2token)
            return train_data, test_data, valid_data, input_size
        if test is None and validation is not None:
            x_valid = vec.transform(validation)
            valid_data = datasets.BOWDataset(x_valid.toarray(), idx2token)
            return train_data, valid_data, input_size
        if test is not None and validation is None:
            x_test = vec.transform(test)
            test_data = datasets.BOWDataset(x_test.toarray(), idx2token)
            return train_data, test_data, input_size
        if test is None and validation is None:
            return train_data, input_size


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    from octis.dataset.dataset import Dataset
    dataset = Dataset()
    dataset.fetch_dataset("20NewsGroup")
    train, validation, test = dataset.get_partitioned_corpus(use_validation=True)
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

    data_corpus_train = [' '.join(i) for i in train]
    data_corpus_test = [' '.join(i) for i in test]
    data_corpus_validation = [' '.join(i) for i in validation]

    vocab = dataset.get_vocabulary()
    x_train, x_test, x_valid, input_size = \
        tool().preprocess(vocab, data_corpus_train, test=data_corpus_test,
                        validation=data_corpus_validation)

    vocab_size = len(vocab)
    model = ProdLDA(
        None,vocab_size, 256, NUM_CLASSES,
        0.2, True,device=device)
    server = NewsProdLDAServer(args, config, model,data_set=dataset,device=device)

    list_models, list_users = server.init_clients()
    server.init_test_loader(x_test)
    model.set_train_data(x_train)

    import numpy as np

    train_data = x_train

    sampler = FLSampler(get_indices_list(), MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                        NUM_CLIENTS)


    print("Sampler initialized")

    from bases.vision.data_loader import DataLoader
    train_loader = DataLoader(train_data, batch_size=CLIENT_BATCH_SIZE, shuffle=False,sampler=sampler,num_workers=0, pin_memory=True)


    client_list = [NewsProdLDAClient(list_models[idx], config, args.use_adaptive,args,device=device) for idx in range(NUM_CLIENTS)]
    for client in client_list:
        client.init_optimizer()
        client.init_train_loader(train_loader)
        client.init_test_loader(x_test)


    print("All initialized. Experiment is {}. Use adaptive = {}.  Client selection = {}. "
          "Num users = {}. Seed = {}. Max round = {}. "
          "MAX_DEC= {}. Batch_size = {}. Num_local_updates = {}".format(EXP_NAME, args.use_adaptive, args.client_selection,
                                       config.NUM_CLIENTS, args.seed, MAX_ROUND, config.MAX_DEC_DIFF, config.CLIENT_BATCH_SIZE,config.NUM_LOCAL_UPDATES))


    fl_runner = ProdLDAFL(args, config, server, client_list)
    fl_runner.main()
