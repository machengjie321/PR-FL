import copy

import torch
from timeit import default_timer as timer

from utils.heap_queue import HeapQueue



@torch.no_grad()
def process_layer(layer, layer_prefix, accumulate_weight: dict, coeff, dtp) -> (
    float, float, list, torch.Tensor):  # 仅处理一层layers
    """

        :param layer: model.layer
        :param layer_prefix:such as ['feauture.0']
        :param accumulate_weight: the acuuumulate_weight of layer, prepare for restoring the model  to a different density
        :param coeff:
        :param dtp:such as 0.3, the model will be pruned to density=0.3 according to the weights
        :return:
        sorted_tba_values.tolist(): the weight value of the pruned neurons in this layer, in decreasing order
        sorted_tba_indices: the idx of sorted_tba_values in original layer
    """
    dtp = 1 - dtp
    # layer.prune_by_pct(dtp)
    # w_name = "{}.weight".format(layer_prefix)
    # b_name = "{}.bias".format(layer_prefix)
    # sqg = accumulate_weight[w_name]  # sqg is a list ，store the accumulate grad of w_name.weight layers
    # iu_mask, niu_mask = layer.mask == 1., layer.mask == 0.
    # num_iu, num_niu = iu_mask.sum().item(), niu_mask.sum().item()
    #
    # # Decrease
    # max_dec_num = int(dtp * num_iu)# the maxium number of neurons that need to prune
    # w_iu = layer.weight[iu_mask]  # use magnitude
    # w_thr = torch.sort(torch.abs(w_iu))[0][max_dec_num]  # 获得在layer.mask==1的weight中获得升序排在0.3位置的weight作为阈值
    # tbk_mask = (torch.abs(layer.weight) >= w_thr) * iu_mask  # 在layer.mask==1的layer中权重大于w_thr的设为1，其余的设为0
    # tba_dec_mask = (torch.abs(layer.weight) < w_thr) * iu_mask  # 在layer.mask==1的layer中权重小于w_thr的设为1，其余的设为0，即获得本次被删去的坐标
    #
    # # Increase
    # tba_inc_mask = niu_mask  # 在layer，mask==0的mask坐标
    #
    #
    # #because the original model is a pruned model, if the model's density is 0.5, and the dtp is 0.5, the final density of the model is 0.25.
    # tba_mask = tba_dec_mask + tba_inc_mask  # 即所有的mask==0的mask
    # tba_values, tba_indices = sqg[niu_mask], niu_mask.nonzero()  # 被删去的梯度和被删去的坐标。
    # sorted_tba_values, sort_perm = torch.sort(tba_values, descending=True)  # 被删去的累积梯度降序排序
    # sorted_tba_indices = tba_indices[sort_perm]

    # prune the layers
    # the process of prune the model is separate from the process of obtain the idx and aqg value of pruned neurons.
    w_name = "{}.weight".format(layer_prefix)
    b_name = "{}.bias".format(layer_prefix)
    sqg = accumulate_weight[w_name]
    iu_mask, niu_mask = layer.mask == 1., layer.mask == 0.
    num_iu, num_niu = iu_mask.sum().item(), niu_mask.sum().item()

    # Decrease
    max_dec_num = int(dtp * num_iu)
    w_iu = layer.weight[iu_mask]  # use magnitude
    w_thr = torch.sort(torch.abs(w_iu))[0][max_dec_num]
    # tbk_mask = (torch.abs(layer.weight) >= w_thr) * iu_mask
    tba_dec_mask = (torch.abs(layer.weight) < w_thr) * iu_mask

    # Increase
    tba_inc_mask = niu_mask

    # total_sqg = sqg[tbk_mask].sum().item()
    # if b_name in accumulate_weight.keys():
    #     total_sqg += accumulate_weight[b_name].sum().item()
    # total_time = coeff * tbk_mask.sum().item()
    tba_mask = tba_dec_mask + tba_inc_mask
    tba_values, tba_indices = sqg[tba_mask], tba_mask.nonzero()
    sorted_tba_values, sort_perm = torch.sort(tba_values, descending=True)
    sorted_tba_indices = tba_indices[sort_perm]

    layer.prune_by_pct(dtp)
    return sorted_tba_values, sorted_tba_indices #the sqg of the decending sorted and idx


def process_layer2(layer, layer_prefix, sgrad: dict, coeff, dtp) -> (float, float, list, torch.Tensor):#仅处理一层layers
    dtp = 1 - dtp
    w_name = "{}.weight".format(layer_prefix)
    b_name = "{}.bias".format(layer_prefix)
    sqg = sgrad[w_name]#sqg is a list ，store the accumulate grad of w_name.weight layers
    iu_mask, niu_mask = layer.mask == 1., layer.mask == 0.
    num_iu, num_niu = iu_mask.sum().item(), niu_mask.sum().item()

    # Decrease
    max_dec_num = int(dtp * num_iu)
    w_iu = layer.weight[iu_mask]  # use magnitude
    w_thr = torch.sort(torch.abs(w_iu))[0][max_dec_num]#获得在layer.mask==1的weight中获得升序排在0.3位置的weight作为阈值
    tbk_mask = (torch.abs(layer.weight) >= w_thr) * iu_mask#在layer.mask==1的layer中权重大于w_thr的设为1，其余的设为0
    tba_dec_mask = (torch.abs(layer.weight) < w_thr) * iu_mask#在layer.mask==1的layer中权重小于w_thr的设为1，其余的设为0，即获得本次被删去的坐标

    # Increase
    tba_inc_mask = niu_mask#在layer，mask==0的mask坐标

    total_sqg = sqg[tbk_mask].sum().item()#保留下来的layer的总数量
    if b_name in sgrad.keys():
        total_sqg += sgrad[b_name].sum().item()#如果bias中也存在积累梯度，也将其算到总数量中
    total_time = coeff * tbk_mask.sum().item()#将未剪枝的layer的总数量*系数

    tba_mask = tba_dec_mask + tba_inc_mask#即所有的mask==0的mask
    tba_values, tba_indices = sqg[tba_mask], tba_mask.nonzero()#被删去的梯度和被删去的坐标。
    sorted_tba_values, sort_perm = torch.sort(tba_values, descending=True)#被删去的累积梯度降序排序
    sorted_tba_indices = tba_indices[sort_perm]

    layer.prune_by_pct(dtp)#prune the layers



    return total_sqg, total_time, sorted_tba_values, sorted_tba_indices

# k-way merge sort
def k_way_merge_sort(tensors):
    if len(tensors) == 1:
        return tensors[0]
    mid = len(tensors) // 2
    left_sorted = k_way_merge_sort(tensors[:mid])
    right_sorted = k_way_merge_sort(tensors[mid:])
    merged = torch.cat((left_sorted, right_sorted))
    return merged[merged[:, 0].argsort()]



def sub_architecture_search(model,accumulate_weight_dict, sum_sqg, sum_time, list_coefficient,list_client_coeff,holistic_coeff, list_tba_values, list_tba_indices,max_density=None):


    list_len = [len(tba) for tba in list_tba_values]
    list_iter = [iter(tba) for tba in list_tba_values]#被删去的sqg值
    # number of params to be added/removed
    list_n = [0] * len(list_len)
    heap_list = []

    heap = HeapQueue([(index, next(_iter) / _coeff) for index, (_iter, _coeff, _len) in
                      enumerate(zip(list_iter, list_coefficient, list_len)) if _len > 0])
    heap_list.append(heap)

    numerator = sum_sqg
    denominator = sum_time


    num_params, max_num = 0, None
    if max_density is not None:
        num_params, max_num = model.calc_num_prunable_params(False)
        max_num = int(max_num * max_density)

    device = model.prunable_layers[0].mask.device
    total_param_in_use = 0
    total_all_param = 0
    for layer, layer_prefx in zip(model.prunable_layers, model.prunable_layer_prefixes):
        layer_param_in_use = layer.num_weight
        layer_all_param = layer.mask.nelement()
        total_param_in_use += layer_param_in_use
        total_all_param += layer_all_param
    list_state_dict = []
    list_sparse_state_dict = []
    list_mask = []
    index = 0
    piu  = total_param_in_use
    ap = total_all_param
    copy_model = copy.deepcopy(model)
    copy_list_n = copy.deepcopy(list_n)
    list_sum_mask = []
    # 尝试对排序过程进行改进
    tensor_tba_values = []
    for index in range(len(list_tba_values)):

        tensor1 = list_tba_values[index] / list_coefficient[index]

        tensor_tba_values.append(torch.cat((tensor1.unsqueeze(1), (torch.ones_like(tensor1, device=device)*index).unsqueeze(1)), dim=1))

    sorted_tensor = k_way_merge_sort(tensor_tba_values)
    sorted_tensor = torch.flip(sorted_tensor, dims=[0])
    index_tensor = sorted_tensor[:, 1]

    # This step is to calculate the masks for different parts of the model.
    for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        target_density = i
        list_n_begin = copy.deepcopy(copy_list_n)

        incre = int(target_density * ap - piu)
        if incre < 0: incre = 0
        incre_index = index_tensor[index:index + incre]
        for pos in range(len(list_tba_values)):
            copy_list_n[pos] += (incre_index == pos).sum()
        index += incre
        piu += incre

        for layer, tba_indices, tba_begin, tba_n in zip(copy_model.prunable_layers, list_tba_indices, list_n_begin, copy_list_n):
            layer.mask[tba_indices[tba_begin:tba_n].t().tolist()] = 1.

        clean_state_dict = copy.deepcopy(copy_model.state_dict())
        mask_state = {}
        for layer, prefix in zip(copy_model.prunable_layers, copy_model.prunable_layer_prefixes):
            # works for both layers
            key_w = prefix + ".weight"
            if key_w in clean_state_dict.keys():
                w_mask = copy_model.get_mask_by_name(key_w)
                mask_state[key_w] = w_mask

        # list_sum_mask.append(copy.deepcopy(mask_state))
        list_sum_mask.append(mask_state)


    sorted_list_client_coeff, sorted_list_client_coeff_idx = torch.sort(
        torch.tensor(list_client_coeff), descending=True)
    lenth = len(list_client_coeff)
    density = []
    begin = sum(list_n)

    begin = 0
    for i in range(lenth):
        list_n_begin = copy.deepcopy(list_n)
        end_condition = False

        while not end_condition:
            obj_val = numerator / denominator

            if begin>len(sorted_tensor):
                print("Exceeds max num")
                break
            val, pos = sorted_tensor[begin]

            if holistic_coeff * val / (sorted_list_client_coeff[i]**2) > obj_val:
                if max_num is not None:
                    if num_params > max_num:
                        print("Exceeds max num")
                        break
                    else:
                        num_params += 1
                begin = begin+1
                pos = int(pos)
                coeff = list_coefficient[pos]

                numerator += val*coeff
                denominator += coeff
                list_n[pos] += 1


            else:
                end_condition = True



        for layer, tba_indices, tba_begin, tba_n in zip(model.prunable_layers, list_tba_indices, list_n_begin, list_n):
            layer.mask[tba_indices[tba_begin:tba_n].t().tolist()] = 1.

        density.append(model.density())

        clean_state_dict = copy.deepcopy(model.state_dict())

        mask_state = {}
        for layer, prefix in zip(model.prunable_layers, model.prunable_layer_prefixes):
            # works for both layers

            key_w = prefix + ".weight"
            if key_w in clean_state_dict.keys():
                weight = clean_state_dict[key_w]
                w_mask = model.get_mask_by_name(key_w)
                real_weight = (weight * w_mask)
                clean_state_dict[key_w] = real_weight
                mask_state[key_w] = w_mask

        for key, value in clean_state_dict.items():
            clean_state_dict[key] = clean_state_dict[key].cpu()

        if i == 0:
            # list_sparse_state_dict.append(copy.deepcopy(clean_state_dict))
            list_sparse_state_dict.append(clean_state_dict)
        else:
            sd = {}
            for key in model.state_dict().keys():
                sd[key] = (clean_state_dict[key] - last_clean_state_dict[key])
            list_sparse_state_dict.append(sd)

        last_clean_state_dict = clean_state_dict
        list_state_dict.append(clean_state_dict)
        # list_mask.append(copy.deepcopy(mask_state))
        list_mask.append(mask_state)



    model_idx = [0] * len(sorted_list_client_coeff_idx)

    for idx in range(len(model_idx)):
        model_idx[sorted_list_client_coeff_idx[idx]] = [i for i in range(idx+1)]

    client_density = []
    list_sd = []
    list_mk = []
    for idx in range(len(model_idx)):
        client_density.append(density[model_idx[idx][-1]])
        list_sd.append(list_state_dict[model_idx[idx][-1]])
        list_mk.append(list_mask[model_idx[idx][-1]])
    model.recover_model()
    return list_sd, model_idx, list_mk, list_sum_mask, list_sparse_state_dict,client_density


@torch.no_grad()
def sub_architecture_search_fast(model,list_coefficient, list_tba_values, list_tba_indices, client_density:list,
                            max_density=None,use_coeff = False):
    """
    Restore the model to different densities according to accumulate_weight

    :param model: server.model, After sub fedavg
    :param list_coefficient: the coefficient of layer, Following the cofficient in Prune_FL, referring to the time used for each layer
    :param list_tba_values:the pruned value of each layer, decreasing sort
    :param list_tba_indices:the list of idx for list_tba_values
    :param client_density:the target density for every client
    :param max_density:

    :return:
    list_state_dict:[[0,1,2,3],[4,5],[6,7],[8],[9,10]]
    model_idx: [[0,1,2],[0],[0,1],[0,1,2,3],[0,1,2,3,4]],but then i need to transform into simluate_server_to_client.client_recv_list

    # """
    # list_len = [len(tba) for tba in list_tba_values]
    # list_iter = [iter(tba) for tba in list_tba_values]  # 被删去的sqg值所组成的列表。len(list_iter) = len(prunable_layers)
    # number of params to be added/removed
    device = model.prunable_layers[0].mask.device
    cpu_device = torch.device("cpu")
    sort_time = timer()
    list_n = [0] * len(list_tba_values)
    sqg_index = []
    # 尝试对排序过程进行改进
    tensor_tba_values = []
    for index in range(len(list_tba_values)):
        if use_coeff:
            tensor1 = list_tba_values[index] / list_coefficient[index]
        else:
            tensor1 = list_tba_values[index]
        tensor_tba_values.append(torch.cat((tensor1.unsqueeze(1), (torch.ones_like(tensor1, device=device)*index).unsqueeze(1)), dim=1))

    sorted_tensor = k_way_merge_sort(tensor_tba_values)
    sorted_tensor = torch.flip(sorted_tensor, dims=[0])
    index_tensor = sorted_tensor[:, 1]

    sort_time = timer() - sort_time


    sorted_clientdensity, sorted_clientdensity_indics = torch.sort(
        torch.tensor(client_density, device=device), descending=False)



    total_param_in_use = 0
    total_all_param = 0
    for layer, layer_prefx in zip(model.prunable_layers, model.prunable_layer_prefixes):
        layer_param_in_use = layer.num_weight
        layer_all_param = layer.mask.nelement()
        total_param_in_use += layer_param_in_use
        total_all_param += layer_all_param
    list_state_dict = []
    list_sparse_state_dict = []
    list_mask = []

    sub_model_time = []


    index = 0
    begin_time = timer()
    piu  = total_param_in_use
    ap = total_all_param
    copy_model = copy.deepcopy(model)
    copy_list_n = copy.deepcopy(list_n)
    list_sum_mask = []

    # This step is to calculate the masks for different parts of the model.
    for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        target_density = i
        list_n_begin = copy.deepcopy(copy_list_n)

        incre = int(target_density * ap - piu)
        if incre < 0: incre = 0
        incre_index = index_tensor[index:index + incre]
        for pos in range(len(list_tba_values)):
            copy_list_n[pos] += (incre_index == pos).sum()
        index += incre
        piu += incre

        for layer, tba_indices, tba_begin, tba_n in zip(copy_model.prunable_layers, list_tba_indices, list_n_begin, copy_list_n):
            layer.mask[tba_indices[tba_begin:tba_n].t().tolist()] = 1.

        clean_state_dict = copy.deepcopy(copy_model.state_dict())
        mask_state = {}
        for layer, prefix in zip(copy_model.prunable_layers, copy_model.prunable_layer_prefixes):
            # works for both layers
            key_w = prefix + ".weight"
            if key_w in clean_state_dict.keys():
                w_mask = copy_model.get_mask_by_name(key_w)
                mask_state[key_w] = w_mask

        # list_sum_mask.append(copy.deepcopy(mask_state))
        list_sum_mask.append(mask_state)
    density = []
    index = 0
    for i in range(len(sorted_clientdensity)):

        target_density = sorted_clientdensity[i]
        list_n_begin = copy.deepcopy(list_n)

        incre = int(target_density*total_all_param - total_param_in_use)
        if incre <0 : incre = 0
        incre_index = index_tensor[index:index+incre]
        for pos in range(len(list_tba_values)):
            list_n[pos] += (incre_index == pos).sum()
        index += incre
        total_param_in_use += incre



        for layer, tba_indices, tba_begin, tba_n in zip(model.prunable_layers, list_tba_indices, list_n_begin, list_n):
            layer.mask[tba_indices[tba_begin:tba_n].t().tolist()] = 1.



        density.append(model.density())

        clean_state_dict = copy.deepcopy(model.state_dict())

        mask_state = {}
        for layer, prefix in zip(model.prunable_layers, model.prunable_layer_prefixes):
            # works for both layers
            key_w = prefix + ".weight"
            if key_w in clean_state_dict.keys():
                weight = clean_state_dict[key_w]
                w_mask = model.get_mask_by_name(key_w)
                real_weight = (weight * w_mask)
                clean_state_dict[key_w] = real_weight
                mask_state[key_w] = w_mask

        for key, value in clean_state_dict.items():
            clean_state_dict[key] = clean_state_dict[key].cpu()

        if i == 0:
            # list_sparse_state_dict.append(copy.deepcopy(clean_state_dict))
            list_sparse_state_dict.append(clean_state_dict)
        else:
            sd = {}
            for key in model.state_dict().keys():
                sd[key] = (clean_state_dict[key] - last_clean_state_dict[key])
            list_sparse_state_dict.append(sd)

        last_clean_state_dict = clean_state_dict
        list_state_dict.append(clean_state_dict)
        list_mask.append(copy.deepcopy(mask_state))
        # list_mask.append(mask_state)

        sub_model_time.append(timer()-begin_time)

    model_idx = [0] * len(sorted_clientdensity)

    for idx in range(len(model_idx)):
        model_idx[sorted_clientdensity_indics[idx]] = [i for i in range(idx+1)]
    client_density = []
    list_sd = []
    list_mk = []
    for idx in range(len(model_idx)):
        client_density.append(round(density[model_idx[idx][-1]], 2))
        list_sd.append(list_state_dict[model_idx[idx][-1]])
        list_mk.append(list_mask[model_idx[idx][-1]])
    model.recover_model()
    return list_sd, model_idx, sort_time, sub_model_time, list_mk, list_sum_mask, list_sparse_state_dict, client_density

@torch.no_grad()
def sub_control(model, accumulate_weight_dict: dict, config,args, client_density:list, min_density,use_coeff = False):
    sum_sqg = 0
    sum_time = config.TIME_CONSTANT
    list_tba_values, list_tba_indices = [], []
    list_coefficient = []

    proc_start = timer()
    comp_coeff_iter = iter(config.COMP_COEFFICIENTS)
    comm_coeff = config.COMM_COEFFICIENT
    for layer, layer_prefix in zip(model.param_layers, model.param_layer_prefixes):
        if layer_prefix in model.prunable_layer_prefixes:

            if use_coeff:
                coeff = comm_coeff + next(comp_coeff_iter)#因为对于这个conv2模型是由4个子模块拼接起来的，所以系数只有4个
            else:
                coeff = 1.0

            sqg, time, sorted_tba_values, sorted_tba_indices = process_layer2(layer, layer_prefix, accumulate_weight_dict,
                                                                             coeff, min_density)
            sum_sqg += sqg
            sum_time += time

            list_coefficient.append(coeff)
            list_tba_values.append(sorted_tba_values)
            list_tba_indices.append(sorted_tba_indices)
        else:
            w_name = "{}.weight".format(layer_prefix)
            b_name = "{}.bias".format(layer_prefix)
            sqg = accumulate_weight_dict[w_name]
            sum_sqg += sqg.sum().item()
            if b_name in accumulate_weight_dict.keys():
                sum_sqg += accumulate_weight_dict[b_name].sum().item()

    process_layers_time = timer() - proc_start


    list_state_dict, model_idx,list_mask,list_sum_mask,list_sparse_state_dict,density = \
        sub_architecture_search(model,accumulate_weight_dict, sum_sqg, sum_time, list_coefficient,
                                args.list_client_coeff,args.holistic_coeff,list_tba_values,list_tba_indices,None)


    sub_model_time = 0

    return list_state_dict, model_idx, sub_model_time, list_mask, list_sum_mask,list_sparse_state_dict, density

@torch.no_grad()
def sub_control_fast(model, accumulate_weight_dict: dict, config, client_density:list, min_density,use_coeff = False):
    sum_sqg = 0
    sum_time = config.TIME_CONSTANT
    list_tba_values, list_tba_indices = [], []
    list_coefficient = []

    proc_start = timer()
    comp_coeff_iter = iter(config.COMP_COEFFICIENTS)
    comm_coeff = config.COMM_COEFFICIENT
    for layer, layer_prefix in zip(model.param_layers, model.param_layer_prefixes):
        if layer_prefix in model.prunable_layer_prefixes:
            coeff = comm_coeff + next(comp_coeff_iter)  # 因为对于这个conv2模型是由4个子模块拼接起来的，所以系数只有4个
            sorted_tba_values, sorted_tba_indices = process_layer(layer, layer_prefix, accumulate_weight_dict,
                                                                             coeff, min_density)


            list_coefficient.append(coeff)
            list_tba_values.append(sorted_tba_values)
            list_tba_indices.append(sorted_tba_indices)
        else:
            w_name = "{}.weight".format(layer_prefix)
            b_name = "{}.bias".format(layer_prefix)

    process_layers_time = timer() - proc_start


    list_state_dict, model_idx, sort_time, sub_model_time, list_mask,list_sum_mask,list_sparse_state_dict,density = sub_architecture_search_fast(model, list_coefficient, list_tba_values, list_tba_indices, client_density,use_coeff=use_coeff)
    sub_model_time = [x+process_layers_time+sort_time for x in sub_model_time]

    return list_state_dict, model_idx, sub_model_time, list_mask, list_sum_mask,list_sparse_state_dict,density

from utils.functional import disp_num_params
def random_prune(model,client_density):
    device = model.prunable_layers[0].mask.device
    sorted_clientdensity, sorted_clientdensity_indics = torch.sort(
        torch.tensor(client_density, device=device), descending=True)
    prune_model = copy.deepcopy(model)
    current_density = disp_num_params(prune_model)
    list_state_dict = []
    list_mask = []
    end_client_density = []
    for density in sorted_clientdensity:

        prune_model = prune_model.random_prune_by_pct(float(1-density/current_density[0]))
        end_client_density.append(disp_num_params(prune_model))
        current_density = disp_num_params(prune_model)

        mask_state = {}
        clean_state_dict = copy.deepcopy(prune_model.state_dict())
        for layer, prefix in zip(model.prunable_layers, model.prunable_layer_prefixes):
            # works for both layers
            key_w = prefix + ".weight"
            if key_w in clean_state_dict.keys():
                weight = clean_state_dict[key_w]
                w_mask = prune_model.get_mask_by_name(key_w)
                real_weight = (weight * w_mask)
                clean_state_dict[key_w] = real_weight
                mask_state[key_w] = w_mask
        list_state_dict.append(copy.deepcopy(clean_state_dict))
        list_mask.append(copy.deepcopy(mask_state))
    model_idx = [0] * len(sorted_clientdensity)



    for idx in range(len(model_idx)):
        model_idx[sorted_clientdensity_indics[idx]] = [i for i in range(idx + 1)]

    return list_state_dict, model_idx,  list_mask





class ControlModule:
    def __init__(self, model, config, args=None):
        self.model = model
        self.config = config
        self.args = args
        self.accumulate_weight_dict = dict()
        self.old_model = None
        self.g = dict()
        self.g_min = dict()


    @torch.no_grad()
    #i have intended to use the average of the weights of the last ten rounds if the server models as one of the evalution metrics for
    #the importance of each part of the model. however, this function need a lot of storge to store the model of the last ten rounds,
    #so i used an approximation to implement this function, When a new round, i will discard 1/10 of the past information of the model
    # and add 1/10 of latest model in deep learning. where 1/10 is similar to the concept of the learning rate.
    def accumulate(self, old_model,i, memory):
        #i: round i;
        for key in self.model.state_dict().keys():
            self.g[key] = self.model.state_dict()[key] - old_model[key]
            if i == 0 :
                self.accumulate_weight_dict[key] = torch.abs(self.g[key]+0)
            if i < memory:
                self.accumulate_weight_dict[key] = (self.accumulate_weight_dict[key]*i+torch.abs(self.g[key]))/(i+1)
            else:
                self.accumulate_weight_dict[key] = (self.accumulate_weight_dict[key]*(memory-1)+self.g[key])/memory



    def rescale_Im(self, Im_dict, scale=1e10, shift=1.5):
        # 将所有的 Im 值合并到一个列表中
        # Step 1: 找到全局最小和最大值
        all_values = torch.cat([v.view(-1) for v in Im_dict.values()])
        global_min = all_values.min()
        global_max = all_values.max()

        normalized_Im_dict = {}

        # Step 2: 根据全局最大最小值归一化每个 tensor
        for k, v in Im_dict.items():
            normalized_Im_dict[k] = (v - global_min) / (global_max - global_min)

        # Step 3: 根据需要的 scale 和 shift 调整范围
        for k, v in normalized_Im_dict.items():
            normalized_Im_dict[k] = v * scale + shift

        return Im_dict



    @torch.no_grad()
    def accumulate_wg(self, sgrad_to_upload, idx, memory):
        for layer, layer_prefix in zip(self.model.param_layers, self.model.param_layer_prefixes):
            if layer_prefix in self.model.prunable_layer_prefixes:
                w_name = "{}.weight".format(layer_prefix)
                b_name = "{}.bias".format(layer_prefix)

                for key in [w_name, b_name]:
                    if key not in sgrad_to_upload.keys():
                        continue
                    if key not in self.accumulate_weight_dict.keys():
                        self.accumulate_weight_dict[key] = sgrad_to_upload[key] * torch.square(self.model.state_dict()[key])
                    else:
                        mask2 = sgrad_to_upload[key] != 0
                        self.accumulate_weight_dict[key][mask2] = sgrad_to_upload[key][mask2] * torch.square(self.model.state_dict()[key][mask2])

                    mask = self.accumulate_weight_dict[key] == 0
                    mask2 = self.accumulate_weight_dict[key] != 0
                    if len(sgrad_to_upload[key][mask2]) != 0:
                        self.g_min[key] = sgrad_to_upload[key][mask2].min()
                    else:
                        self.g_min[key] = min(self.g_min.values())
                    if len(self.accumulate_weight_dict[key][mask]) != 0:
                        self.accumulate_weight_dict[key][mask] = self.g_min[key] * torch.abs(self.model.state_dict()[key][mask])



    @torch.no_grad()
    def accumulate_g(self,sgrad_to_upload):
        for layer, layer_prefix in zip(self.model.param_layers, self.model.param_layer_prefixes):
            if layer_prefix in self.model.prunable_layer_prefixes:
                w_name = "{}.weight".format(layer_prefix)
                b_name = "{}.bias".format(layer_prefix)

                for key in [w_name,b_name]:
                    if key not in sgrad_to_upload.keys():
                        continue
                    if key not in self.accumulate_weight_dict.keys():
                        self.accumulate_weight_dict[key] = sgrad_to_upload[key]
                        continue
                    mask2 = sgrad_to_upload[key] != 0
                    self.accumulate_weight_dict[key][mask2] = sgrad_to_upload[key][mask2]

                    mask = self.accumulate_weight_dict[key] == 0
                    mask2 = self.accumulate_weight_dict[key] != 0
                    if len(sgrad_to_upload[key][mask2]) != 0:
                        self.g_min[key] = sgrad_to_upload[key][mask2].min()
                    else:
                        self.g_min[key] = min(self.g_min.values())

                    if len(self.accumulate_weight_dict[key][mask]) != 0:
                        self.accumulate_weight_dict[key][mask] = self.g_min[key] * torch.abs(self.model.state_dict()[key][mask])




    @torch.no_grad()
    def sub_adjust(self, client_density: list, use_coff=None, min_density=None):
        list_state_dict, model_idx, sub_model_time, list_mask = sub_control(self.model, self.accumulate_weight_dict, self.config, client_density, min_density)
        return list_state_dict, model_idx, sub_model_time,  list_mask

    @torch.no_grad()
    def sub_adjust_fast(self, client_density: list, use_coff = None, min_density=None):
        list_state_dict, model_idx, sub_model_time, list_mask, list_sum_mask, list_sparse_state_dict,density = sub_control_fast(self.model, self.accumulate_weight_dict,
                                                                            self.config, client_density, min_density,use_coff)
        return list_state_dict, model_idx, sub_model_time, list_mask, list_sum_mask, list_sparse_state_dict,density

    @torch.no_grad()
    def sub_adjust_prune(self, client_density: list, use_coff=None, min_density=0.1):

        list_state_dict, model_idx, sub_model_time, list_mask, list_sum_mask, list_sparse_state_dict,density = sub_control(self.model, self.accumulate_weight_dict,
                                                                            self.config, self.args, client_density, min_density,use_coeff=use_coff)

        return list_state_dict, model_idx, sub_model_time, list_mask, list_sum_mask, list_sparse_state_dict,density



import heapq

@torch.no_grad()
def simulate_client_to_server(time, size, upload_speed, server_download_speed):
    # 客户端数量
    n = len(time)
    # 用于存储每个客户端发送文件的状态
    # state[i] = 0 表示第i个客户端还没有开始发送文件
    # state[i] = 1 表示第i个客户端正在发送文件但是还没有发送完成
    # state[i] = 2 表示第i个客户端已经发送完成但是服务器还没有接收完成
    # state[i] = 3 表示第i个客户端的文件已经被服务器接收完成
    state = [0] * n

    # 考虑每个客户端开始发送文件和文件全部上传的时间为上传总流量的变化时间点
    min_time_heap = []
    server_receive_time = [0] * n
    # print(time)
    # print(size)
    # print(upload_speed)
    for i in range(n):
        heapq.heappush(min_time_heap, time[i])
        heapq.heappush(min_time_heap, time[i] + size[i] / upload_speed[i])

    next_time = heapq.heappop(min_time_heap)
    current_time = next_time
    # 存储每个客户端上传到网络的文件大小
    uploaded_size = [0] * n

    next_time = heapq.heappop(min_time_heap)

    client_sequence = []

    # 只有当所有客户端的文件都已经被服务器接收才退出循环
    while True:
        # 存储每个客户端本次循环上传/下载的大小
        cycle_size = [0] * n
        # 遍历每个客户端，计算本次时间段上传文件的总大小
        for i in range(n):
            # 如果第i个客户端已经发送完成并且服务器已经接收完成，则跳过
            if state[i] == 3:
                continue
            # 如果第i个客户端还没有开始发送文件，则判断当前时间是否已经到达发送时间
            if state[i] == 0:
                if current_time == time[i]:
                    state[i] = 1
                    client_sequence.append(i)
            if state[i] == 1:
                cycle_size[i] = min(upload_speed[i] * (next_time - current_time), size[i])
                uploaded_size[i] += cycle_size[i]
                size[i] -= cycle_size[i]
                if size[i] < 1e-9: #if i use == 0, bugs may occur due to errors in high-precision calculation
                    state[i] = 2

        # 模拟服务器下载文件的过程

        max_download_size = server_download_speed * (next_time - current_time)

        for j in client_sequence:
            if max_download_size >= uploaded_size[j]:
                max_download_size -= uploaded_size[j]
                uploaded_size[j] = 0
                if state[j] == 2:
                    state[j] = 3
                    server_receive_time[j] = next_time


            else:
                uploaded_size[j] -= max_download_size

                break

        current_time = next_time

        if len(min_time_heap) > 0:
            next_time = heapq.heappop(min_time_heap)
        else:

            next_time = current_time + max(((sum(uploaded_size) + sum(size)) / server_download_speed), 0.001)

        if sum(state) == n * 3:
            break
        # else:
        #     print(state)
        #     print(size)
        #     print(next_time,current_time)

    return server_receive_time

def determine_density(server_recive_time):
    client_density = []
    return client_density

import heapq

@torch.no_grad()
def simluate_server_to_client(time,client_stare_work_time, size, server_upload_speed, model_idx, client_download_speed):

    client_recv_list = [[0 for _ in range(len(model_idx))] for _ in range(len(model_idx))]
    for client_idx in range(len(model_idx)):
        for file_idx in model_idx[client_idx]:
            client_recv_list[file_idx][client_idx] = 1



    # 客户端数目
    n = len(client_download_speed)
    client_receive_time = [-1 for _ in range(n)]
    # 文件数目
    m = len(time)
    import numpy as np
    waiting_recv_number = np.array(client_recv_list).sum(axis=0)

    # 文件
    client_download_size = [[0 for _ in range(n)] for _ in range(m)]
    # 客户端状态
    # 0-未开始接收，1-接收完成
    client_status = [[0 for _ in range(n)] for _ in range(m)]

    # 当前时间
    current_time = 0
    time_increment = 0.05

    time_increment_client_download_bandwidth = [round(i * time_increment, 2) for i in client_download_speed]

    # 初始化每个文件的已上传字节数和上传时间点
    file_uploaded_size = [0] * m
    file_uploaded_time = [-1] * m

    # 0:文件尚未上传;1:文件正在上传;2:文件已经成功上传
    server_upload_state = [0] * m

    time_increment_server_current_bandwidth = server_upload_speed * time_increment
    while True:
        current_time = round(current_time + time_increment, 2)
        current_bandwidth = time_increment_server_current_bandwidth
        # 模拟服务器上传情况，以time_increment作为精度模拟服务器文件上传
        for i in range(m):
            if server_upload_state[i] == 2: continue  # 如果第i个文件已经上传，直接进行下一个文件
            if server_upload_state[i] == 0 and current_time >= time[i]:  # 即第i个文件还没上传但是已经到了要上传的时间了
                server_upload_state[i] = 1
            if server_upload_state[i] == 1 and size[i] >= 0:  # size[i]是第i个文件还未上传的文件
                if current_bandwidth >= size[i]:
                    current_bandwidth = current_bandwidth - size[i]
                    file_uploaded_size[i] += size[i]
                    size[i] = 0
                    server_upload_state[i] = 2
                    file_uploaded_time[i] = current_time

                else:
                    size[i] = size[i] - current_bandwidth
                    file_uploaded_size[i] += current_bandwidth
                    current_bandwidth = 0
                    break


        # 模拟服务器下载情况,以time_increment作为精度模拟服务器文件上传
        client_download_bandwidth = time_increment_client_download_bandwidth.copy()
        for i in range(m):
            for j in range(n):
                # 如果client[j]需要接收文件i
                if current_time <= client_stare_work_time[j]:
                    continue
                if client_recv_list[i][j] == 1:
                    # 如果对于客户端j已经没有剩余的带宽了，则跳过客户端j
                    if client_download_bandwidth[j] == 0: continue
                    # 检查对于client[j]是否已经接收i
                    if client_status[i][j] == 1: continue
                    # 如果文件i还未上传
                    if server_upload_state[i] == 0:
                        break
                    # 如果文件已经上传
                    else:
                        # 如果客户端j当前下载的文件i小于服务器已经上传的文件i
                        if client_download_size[i][j] < file_uploaded_size[i]:
                            c = min(file_uploaded_size[i] - client_download_size[i][j], client_download_bandwidth[j])
                            client_download_bandwidth[j] -= c
                            client_download_size[i][j] += c

                        # 如果服务器的文件已经完全上传，并且客户端下载的文件大小与已经上传的大小相同，则代表该客户端已经完全接收了该文件
                        if server_upload_state[i] == 2 and client_download_size[i][j] >= file_uploaded_size[i]:
                            client_status[i][j] = 1


        client_recved_number = np.array(client_status).sum(axis=0)



        for j in range(n):
            if client_recved_number[j] == waiting_recv_number[j] and client_receive_time[j] == -1:
                client_receive_time[j] = current_time

        if client_recved_number.sum() == waiting_recv_number.sum():
            return client_receive_time
