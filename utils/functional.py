from typing import Union, Generator
from copy import deepcopy
import random

import numpy as np
def disp_num_params(model):
    total_param_in_use = 0
    total_all_param = 0
    for layer, layer_prefx in zip(model.prunable_layers, model.prunable_layer_prefixes):
        layer_param_in_use = layer.num_weight
        layer_all_param = layer.mask.nelement()
        total_param_in_use += layer_param_in_use
        total_all_param += layer_all_param
        print("{} remaining: {}/{} = {}".format(layer_prefx, layer_param_in_use, layer_all_param,
                                                layer_param_in_use / layer_all_param))
    print("Total: {}/{} = {}".format(total_param_in_use, total_all_param, total_param_in_use / total_all_param))

    return total_param_in_use / total_all_param, total_param_in_use


def copy_dict(ori_dict: Union[dict, Generator]):
    generator = ori_dict.items() if isinstance(ori_dict, dict) else ori_dict
    copied_dict = dict()
    for key, param in generator:
        copied_dict[key] = param
    return copied_dict


def deepcopy_dict(ori_dict: Union[dict, Generator]):
    generator = ori_dict.items() if isinstance(ori_dict, dict) else ori_dict
    deepcopied_dict = dict()
    for key, param in generator:
        deepcopied_dict[key] = param.clone()
    return deepcopied_dict


def copy_shuffle_list(inp_list):
    copy_list = deepcopy(inp_list)
    random.shuffle(copy_list)
    return copy_list


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
    client_idcs = sorted(client_idcs, key=lambda x: (len(x)), reverse=False)

    return client_idcs