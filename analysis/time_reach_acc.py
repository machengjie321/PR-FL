import copy

import matplotlib.pyplot as plt
import numpy as np
# Create data
import numpy as np
import os
from os.path import join

from itertools import product
from utils.save_load import load



client_sel = False
def load_model_sum(exp, cs=False):
    return load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), "list_model_sum.pt"))

def load_R2SP_sum(exp, cs=False):
    return load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), "R2SP_client_sum.pt"))

def load_server_to_client_sum(exp, cs=False):
    return load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), "sever_to_client_sum.pt"))

def load_client_loss(exp, cs=False):
    return load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), "list_client_loss.pt"))

def load_client_acc(exp, cs=False):
    return load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), "list_client_acc.pt"))

def load_client_size(exp, cs=False):
    return load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), "list_model_size.pt"))

def load_checkpoind(exp):
    return load(os.path.join(result_path, "{}".format(exp), 'checkpoint.pth'))

def load_acc(exp, cs=False):
    return load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), "accuracy.pt"))

def load_time(exp, cs=False):
    return load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), "est_time.pt"))


def load_ms(exp, cs=False):
    return load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), "model_size.pt"))

def load_fed_avg_acc(exp, cs=False):
    return load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), "fed_avg_acc.pt"))

def load_fed_avg_loss(exp, cs=False):
    return load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), "fed_avg_loss.pt"))

def load_model_G(exp, cs=False):
    return load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), "model_G.pt"))
def load_client_density(exp, cs=False):
    return load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), "self.list_client_density"))
def load_l1_client_weight_sum(exp, cs=False):
    return load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), 'self.list_l1_client_weight_sum'))
def load_l2_client_weight_sum(exp, cs=False):
    return load(join(result_path, "{}{}".format(exp, "_cs" if cs else ""), 'self.list_l2_client_weight_sum'))
if os.getcwd().startswith('/mnt/sda1/mcj/PruneFL-master/PruneFL-master'):
    os.chdir('/mnt/sda1/mcj/PruneFL-master/PruneFL-master')


data1 = [[
    "PIF_CelebA_[1, 0.5, 0.1]n_-1.0_30_0.2_1e-05_sub_fed_avg_g_0.0_3_n_False_True",
    "PIF_CelebA_[1, 0.5, 0.1]n_-0.5_30_0.2_1e-05_sub_fed_avg_g_0.0_3_n_False_True",
    "PIF_CelebA_[1, 0.5, 0.1]n_-1.5_30_0.2_1e-05_sub_fed_avg_g_0.0_3_n_False_True",
    "PIF_CelebA_[1, 0.5, 0.1]n_-1.0_30_0.2_1e-05_sub_fed_avg_g_0.0_3_c_False_True",
    "PIF_CelebA_[1, 0.5, 0.1]n_0.0_30_0.2_1e-05_sub_fed_avg_g_0.0_10_n_False_False",
    "PIF_CelebA_[1, 0.5, 0.1]n_0.0_30_0.2_1e-05_sub_fed_avg_g_0.0_3_n_False_True",
    "PIF_CelebA_[1, 0.5, 0.1]n_1.0_30_0.2_1e-05_sub_fed_avg_g_0.0_10_n_False_True",
    "PIF_CelebA_[1, 0.5, 0.1]n_-1.0_30_0.2_1e-05_sub_fed_avg_g_0.0_10_n_False_True",
    "PIF_CelebA_[1, 0.5, 0.1]n_0.0_30_0.2_1e-05_sub_fed_avg_g_0.0_10_n_False_True",
    "PIF_CelebA_[1, 0.5, 0.1]n_0.0_30_0.2_1e-05_sub_fed_avg_g_0.0_10_c_False_True",
    "PIF_CelebA_[1, 0.5, 0.1]n_0.0_30_0.2_0.0_sub_fed_avg_g_0.0_10_n_False_True",
    "CelebA_1",
    "CelebA_1wd"
]]
config = None
len_round ={}
min_modelsize = {}
density_round = {}
end_best_acc={}
end_best_NPMI={}
end_best_td={}
acc_time = {}
npmi_time = {}
td_time = {}
train_time = {}
all_time = {}
best_acc={}
td_max_acc = {}
n = 10
import pandas as pd
experiment_data = {
    'exp_name': [],
    'best_acc_index': [],
    'best_acc_value': [],
    'td_max_acc_1': [],
    'td_max_acc_2': [],
    'td_max_acc_3': [],
    'td_max_acc_4': [],
    'len_round': [],
    'density_round': [],
    'end_best_acc': [],

}
for data in data1:
    if 'CelebA' in data[0]:
        import configs.celeba as config
        td = [500,1000,1500,1800]
    if 'CIFAR10' in data[0]:
        import configs.cifar10 as config

    if 'TINYIMAGENET' in data[0]:
        import configs.imagenet100 as config

    if 'femnist' in data[0]:
        import configs.femnist as config

    result_path = join("results", config.EXP_NAME)
    for exp_name in data:
        experiment_data['exp_name'].append(exp_name)
        acc = np.array(load_fed_avg_acc(exp_name))
        max_acc = np.max(acc)
        best_acc[exp_name] = [(np.argmax(acc)-1)*config.EVAL_DISP_INTERVAL,np.max(acc)]
        experiment_data['best_acc_index'].append(best_acc[exp_name][0])
        experiment_data['best_acc_value'].append(best_acc[exp_name][1])
        tdmax_acc = []
        for d in td:
            n = d//config.EVAL_DISP_INTERVAL+1
            if (n+5) > len(acc)-1:
                c = len(acc-1)
            else:
                c = n+5
            if n-5 >= c:
                m = c-1
            else:
                m = n-5
            tdmax_acc.append([d,np.max(acc[m:c])])
        td_max_acc[exp_name] = copy.deepcopy(tdmax_acc)
        experiment_data['td_max_acc_1'].append(td_max_acc[exp_name][0][1])
        experiment_data['td_max_acc_2'].append(td_max_acc[exp_name][1][1])
        experiment_data['td_max_acc_3'].append(td_max_acc[exp_name][2][1])
        experiment_data['td_max_acc_4'].append(td_max_acc[exp_name][3][1])
        len_round[exp_name] = (len(acc)-1)*config.EVAL_DISP_INTERVAL+1
        experiment_data['len_round'].append(len_round[exp_name])


        try:
            client_density = np.array(load_client_density(exp_name))[:, -1]
            client_density = client_density[::config.EVAL_DISP_INTERVAL]
            assert len(client_density) == len(acc)
            end_best_acc[exp_name] = np.max(acc[client_density==1])
            dr = []
            for index in range(1, len(client_density)):
                if client_density[index] != client_density[index - 1]:
                    dr.append([round(client_density[index],1),(index-1)*config.EVAL_DISP_INTERVAL])
            density_round[exp_name] = copy.deepcopy(dr)



        except FileNotFoundError:
            print(f"do not have client_density, {exp_name}")
        experiment_data['density_round'].append(density_round.get(exp_name, []))
        experiment_data['end_best_acc'].append(end_best_acc.get(exp_name,[]))



import pandas as pd

# 您提供的代码


df = pd.DataFrame(experiment_data)

# 指定要保存的Excel文件名
excel_file_path = join("results",'experiment_results.xlsx')


# 将DataFrame写入Excel文件，不包含索引列
df.to_excel(excel_file_path, index=False)









