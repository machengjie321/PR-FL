from torch.utils.data import Sampler
from copy import deepcopy
from typing import List
import numpy as np
from utils.functional import copy_shuffle_list
import random

class FLSampler(Sampler):
    '''
    input:
    indices_partition : at leat the two-dimensional matrix  n*k，the n is the number of client，k is the number of data on the NTH client.
    data_per_client: the number of data required by a client training each round.
    client_per_round: the idx of selected client in each round.if None,all client are selected in each round

    output:
    sequence:the sequence of data to be trained,if the number_round = 1 and client_per_round=None,the length of output is
             data_per_client*the number of client[selected data of client1,selected data of client2,...selected data of clientn]


    '''
    def __init__(self, indices_partition: List[List],idx, num_round, data_per_client, client_selection,
                 client_per_round=None):
        num_partition = len(indices_partition)
        self.sequence = None

        range_partition = list(range(num_partition))
        copy_list_ind = deepcopy(indices_partition)
        new_list_ind = [[] for _ in range(num_partition)]

        if client_selection:
            assert client_per_round is not None
            assert client_per_round <= num_partition

        list_pos = [0] * num_partition
        for rd_idx in range(num_round):
            if client_selection:
                selected_client_idx = random.sample(range_partition, client_per_round)
            else:
                selected_client_idx = range_partition

            for client_idx in selected_client_idx:
                ind = copy_list_ind[client_idx]# 得到了第i个客户端的数据下标
                pos = list_pos[client_idx]
                while len(new_list_ind[client_idx]) < pos + data_per_client:
                    random.shuffle(ind)
                    new_list_ind[client_idx].extend(ind)

                list_pos[client_idx] = pos + data_per_client
        self.sequence = new_list_ind[idx]
        # but it is no difference than directly assigning clients different users


    def __iter__(self):
        return iter(self.sequence)

    def __len__(self):
        return len(self.sequence)


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    n_classes = train_labels.max()+1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs