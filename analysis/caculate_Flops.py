import torch

from experiments.NEWs.results.prodLDA.PruneFL_Prod_LDA import preprocess
from octis.dataset.dataset import Dataset
from bases.vision.data_loader import DataLoader
import torchprofile
from bases.nn.sequential import DenseSequential,SparseSequential
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
num_topics=20
import numpy as np
import os
from os.path import join
import matplotlib.pyplot as plt
from itertools import product
from utils.save_load import load
n = 10
import datetime


result_path = '/mnt/sda1/mcj/PruneFL-master/PruneFL-master/results/prodLDA/'
def load_model(exp):
    model = load(join(result_path,exp,'model.pt'))
    return model
name1 =['1.0','0.8','0.8*4','0.6','0.6*4','0.4','0.4*4','0.2','0.2*4','0.1','0.1*4','0.01','0.01*4']
prefix1 = 'ProdLDA_20NewsGroup_2e-3target_density_'
#prefix ='ProdLDA_M10_2e-3target_density_'
#prefix = 'ProdLDA_BBC_News_2e-3target_density_'
prefix2 = 'ProdLDA_DBLP_2e-3target_density_'
for name in [name1]:
    for prefix in [prefix1,prefix2]:

        #to initialize the exp
        exp = []
        for i in name:
            exp.append(prefix+i)


import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_to_sparse(model):
    for name, param in model.named_parameters():
        if 'weight' in name:  # Convert only weight parameters to sparse
            mask = param != 0  # Create a mask of non-zero elements
            indices = torch.nonzero(mask)
            values = param[mask]
            shape = param.shape

            sparse_param = torch.sparse.FloatTensor(indices.t(), values, shape)
            setattr(model, name, sparse_param)


# Define the first neural network
class Net1(nn.Module):
    def __init__(self,num_topics=20, vocab_size=1513):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features=1513, out_features=100, bias=True),
            nn.Softplus(beta=1, threshold=20)
        )
        hidden_size = (100,100)
        from collections import OrderedDict
        self.hiddens =nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), nn.Softplus()))
            for i, (h_in, h_out) in enumerate(zip(hidden_size[:-1], hidden_size[1:]))]))
        self.f_drop =  nn.Dropout(p=0.2, inplace=False)
        self.f_mu = nn.Sequential(nn.Linear(in_features=100, out_features=20, bias=True),
                                    nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True))
        self.f_sigma =  nn.Sequential(nn.Linear(in_features=100, out_features=20, bias=True),
                                    nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True))
        self.beta_batchnorm = nn.BatchNorm1d(1513, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.drop_theta = nn.Dropout(p=0.2, inplace=False)
        self.beta = nn.Parameter(torch.Tensor(num_topics, vocab_size))
        self.prior_mean = torch.tensor(
            [0.0] * num_topics)
        self.prior_mean = nn.Parameter(self.prior_mean)

        topic_prior_variance = 1. - (1. / num_topics)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * num_topics)
        self.prior_variance = nn.Parameter(self.prior_variance)
        self.topic_word_matrix = nn.Parameter(torch.tensor([1.1,2.2]))
        self.final_topic_word = nn.Parameter(torch.tensor([1.1,2.2]))


    def forward(self, inputs):
        outputs = self.input_layer(inputs)
        outputs = self.hiddens(outputs)

        outputs = self.f_drop(outputs)
        posterior_mu = self.f_mu(outputs)
        posterior_log_sigma = self.f_sigma(outputs)

        posterior_sigma = torch.exp(posterior_log_sigma)

        # generate samples from theta
        theta = F.softmax(
            self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)
        topic_doc = theta
        theta = self.drop_theta(theta)
        # in: batch_size x input_size x n_components
        word_dist = F.softmax(
        self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
        topic_word = self.beta
        # word_dist: batch_size x input_size
        self.topic_word_matrix = self.beta
        self.final_topic_word = topic_word
        self.final_topic_document = topic_doc

        return posterior_mu, posterior_sigma, posterior_log_sigma, word_dist, topic_word, topic_doc



    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)










macs_dict = {}
for exp_name in exp:
    model = load_model(exp_name)
    # model.input_layer = model.input_layer.to_sparse()
    # model.hiddens = model.hiddens[0].to_sparse()
    # model.f_mu = DenseSequential(SparseSequential(model.f_mu[0].to_sparse()),model.f_mu[1])
    #
    # model.f_sigma = DenseSequential(SparseSequential(model.f_sigma[0].to_sparse()),model.f_sigma[1])
    dataset = Dataset()
    if exp_name[0:9] == 'ProdLDA_2':
        dataset.fetch_dataset("20NewsGroup")
    if exp_name[0:9] == 'ProdLDA_D':
        dataset.fetch_dataset("DBLP")

    train, validation, test = dataset.get_partitioned_corpus(use_validation=True)
    data_corpus_train = [' '.join(i) for i in train]
    data_corpus_test = [' '.join(i) for i in test]
    data_corpus_validation = [' '.join(i) for i in validation]
    vocab = dataset.get_vocabulary()
    vocab_size = len(vocab)
    x_train, x_test, x_valid, input_size = \
        preprocess(vocab, data_corpus_train, test=data_corpus_test,
                   validation=data_corpus_validation)

    import numpy as np
    train_data = x_train.X
    shuffle_ix = np.random.permutation(np.arange(len(train_data)))
    train_data = train_data[shuffle_ix]
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=0,
                              pin_memory=True)
    x = train_loader.get_next_batch().float()



    x = x.to(device)

    model.to(device)
    net = Net1()
    net.load_state_dict(model.state_dict())



    macs_dict[exp_name] = torchprofile.profile_macs(model,x)