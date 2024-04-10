from torch.utils.data import Sampler
from copy import deepcopy
from typing import List
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
    def __init__(self, indices_partition: List[List], num_round, data_per_client, client_selection,
                 client_per_round=None):
        self.sequence = []
        num_partition = len(indices_partition)
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
                ind = copy_list_ind[client_idx]
                pos = list_pos[client_idx]
                while len(new_list_ind[client_idx]) < pos + data_per_client:
                    random.shuffle(ind)
                    new_list_ind[client_idx].extend(ind)
                self.sequence.extend(new_list_ind[client_idx][pos:pos + data_per_client])
                list_pos[client_idx] = pos + data_per_client

    def __iter__(self):
        return iter(self.sequence)

    def __len__(self):
        return len(self.sequence)
