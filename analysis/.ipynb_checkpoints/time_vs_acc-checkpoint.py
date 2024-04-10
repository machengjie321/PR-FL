import numpy as np
import os
from os.path import join
import matplotlib.pyplot as plt
from itertools import product
from utils.save_load import load
n = 10
import datetime


def load_acc(exp):
    acc = load(join(result_path, "{}".format(exp), "list_accuracy.pt"))

    return acc

def load_model_size(exp):
    if prefix == 'ProdLDA_20NewsGroup_2e-3target_density_':
        return np.array(load(join(result_path, "{}".format(exp), "model_size.pt")))[
               ::5]
    if prefix == 'ProdLDA_DBLP_2e-3target_density_':
        return np.array(load(join(result_path, "{}".format(exp), "model_size.pt")))[
               ::1]
def load_npmi(exp):
    npmi = load(join(result_path, "{}".format(exp), "npmi.pt"))
    return npmi

def load_topic_diversity(exp):
    td = load(join(result_path, "{}".format(exp), "topic_diversity.pt"))
    return td

def load_time(exp):
    if prefix == 'ProdLDA_20NewsGroup_2e-3target_density_':
        time = np.cumsum(load(join(result_path, "{}".format(exp), "est_time.pt")))[
               ::5]
    if prefix == 'ProdLDA_DBLP_2e-3target_density_':
        time = np.cumsum(load(join(result_path, "{}".format(exp), "est_time.pt")))[
               ::1]
    return time

result_path = '/mnt/sda1/mcj/PruneFL-master/PruneFL-master/results/prodLDA/'
# name_20News = ['ProdLDA_20NewsGroup_2e-3target_density_0.01second',
#                'ProdLDA_20NewsGroup_2e-3target_density_0.1second','ProdLDA_20NewsGroup_2e-3target_density_0.1',
#           'ProdLDA_20NewsGroup_2e-3target_density_0.4*4second', 'ProdLDA_20NewsGroup_2e-3target_density_0.4*4',
#           'ProdLDA_20NewsGroup_2e-3target_density_0.4second', 'ProdLDA_20NewsGroup_2e-3target_density_0.4']
#
# name_DBLP = ['ProdLDA_DBLP_2e-3target_density_0.1*4second', 'ProdLDA_DBLP_2e-3target_density_0.1',
#           'ProdLDA_DBLP_2e-3target_density_0.2*4second','ProdLDA_DBLP_2e-3target_density_0.2',
#           'ProdLDA_DBLP_2e-3target_density_0.4*4second','ProdLDA_DBLP_2e-3target_density_0.4*4']
# #name = ['1.0','0.8','0.8*4','0.6','0.6*4','0.4','0.4*4','0.2','0.2*4','0.1','0.1*4','0.01','0.01*4']
# #name = ['1.0','0.8','0.8*4','0.2','0.2*4']
# #prefix = 'ProdLDAtarget_density_'
# #to initialize the exp
#
# #
# prefix = 'ProdLDA_DBLP_2e-3target_density_'
# # #Time and acc
# for exp_name in name_DBLP:
#     try:
#         acc = load_acc(exp_name)
#         acc = np.convolve(acc, np.ones((n,)) / n, mode='valid')
#
#         time = load_time(exp_name)
#         time = np.convolve(time, np.ones((n,)) / n, mode='valid')
#
#
#
#         plt.plot(time, acc, linewidth=1)
#     except FileNotFoundError:
#         print(f"Skipping training results for {exp_name}. ")
# #
# # if exp_name[0:9] == 'ProdLDA_2':
#     time_lim = (-2500, 20000)
#     acc_lim = (0.2,0.5)
#     plt.axhline(y=0.4, color='red', linestyle='--')
# # if exp_name[0:9] == 'ProdLDA_D':
#     time_lim = (-200, 3000)
#     acc_lim = (0.4, 0.70)
#     plt.axhline(y=0.62, color='red', linestyle='--')
# #
# plt.xlabel(r"Time (s)")
# plt.ylabel("Test Accuracy")
#
#
# plt.xlim(time_lim)
# plt.ylim(acc_lim)
#
# plt.grid(linestyle="--", color='black', lw='0.5', alpha=0.5)
#
# plt.legend(name_DBLP,
#            frameon=False, loc="center right")
# current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# fig_path = join(result_path, "figs")
# os.makedirs(fig_path, exist_ok=True)
# plt.savefig(join(fig_path, 'second'+current_time+"Accuracy"), dpi=300)
# plt.close()


name1 =['1.0','0.8','0.8*4','0.6','0.6*4','0.4','0.4*4','0.2','0.2*4','0.1','0.1*4','0.01','0.01*4']
name2 = ['1.0','0.8','0.6','0.4','0.2','0.1','0.01']
name3 = ['1.0','0.8*4','0.6*4','0.4*4','0.2*4','0.1*4','0.01*4']
name4 =['1.0','0.4*4','0.2','0.1','0.01']
name5 =['1.0','0.8','0.8*4','0.6','0.6*4','0.4','0.4*4','0.2','0.2*4','0.01']
prefix1 = 'ProdLDA_20NewsGroup_2e-3target_density_'
#prefix ='ProdLDA_M10_2e-3target_density_'
#prefix = 'ProdLDA_BBC_News_2e-3target_density_'
prefix2 = 'ProdLDA_DBLP_2e-3target_density_'
for name in [name4]:
    for prefix in [prefix1]:

        #to initialize the exp
        exp = []
        for i in name:
            exp.append(prefix+i)
        # to initialize the legend
        den=[]
        for i in exp:
            if i[len(i)-2:] == '*4':
                if i[len(i)-6:-2] == '0.01':
                    den.append('Fast_'+i[len(i)-6:-2])
                else:
                    den.append('Fast_'+i[len(i)-5:-2])
            else:
                if i[len(i) - 4:] == '0.01':
                    den.append('Normal_'+i[len(i) - 4:])
                else:
                    den.append('Normal_'+i[len(i) - 3:])


        #Time and acc
        for exp_name in exp:
            try:
                acc = load_acc(exp_name)
                acc = np.convolve(acc, np.ones((n,)) / n, mode='valid')

                time = load_time(exp_name)
                time = np.convolve(time, np.ones((n,)) / n, mode='valid')



                plt.plot(time, acc, linewidth=1)
            except FileNotFoundError:
                print(f"Skipping training results for {exp_name}. ")

        if prefix == 'ProdLDA_20NewsGroup_2e-3target_density_':
            n = 10
            time_lim = (-2500, 20000)
            acc_lim = (0.2,0.5)
            plt.axhline(y=0.4, color='red', linestyle='--')
        if prefix == 'ProdLDA_DBLP_2e-3target_density_':
            n = 5
            time_lim = (-200, 3000)
            acc_lim = (0.4, 0.70)
            plt.axhline(y=0.62, color='red', linestyle='--')

        plt.xlabel(r"Time (s)")
        plt.ylabel("Test Accuracy")


        plt.xlim(time_lim)
        plt.ylim(acc_lim)

        plt.grid(linestyle="--", color='black', lw='0.5', alpha=0.5)

        plt.legend(den,
                frameon=False, loc="center right")
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig_path = join(result_path, "figs")
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(join(fig_path, prefix+current_time+"Accuracy"), dpi=300)
        plt.close()

        #Round and acc
        for exp_name in exp:
            try:
                acc = load_acc(exp_name)
                acc = np.convolve(acc, np.ones((n,)) / n, mode='valid')

                x = np.arange(len(acc))
                if prefix == 'ProdLDA_20NewsGroup_2e-3target_density_':
                    x = x*5#because for 20NewsGroup, results are tested every five rounds
                plt.plot(x, acc, linewidth=1)
            except FileNotFoundError:
                print(f"Skipping training results for {exp_name}. ")

        if prefix == 'ProdLDA_20NewsGroup_2e-3target_density_':
                acc_lim = (0.2,0.5)
                x_lim = (-10,2600)
                plt.axhline(y=0.4, color='red', linestyle='--')
        if prefix =='ProdLDA_DBLP_2e-3target_density_':
                acc_lim = (0.4, 0.70)
                x_lim = (-5,450)
                plt.axhline(y=0.62, color='red', linestyle='--')

        plt.xlabel("Rounds")
        plt.ylabel("Test Accuracy")


        plt.xlim(x_lim)
        plt.ylim(acc_lim)

        plt.grid(linestyle="--", color='black', lw='0.5', alpha=0.5)

        plt.legend(den,
                   frameon=False, loc="center right")
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig_path = join(result_path, "figs")
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(join(fig_path, prefix+current_time+"Round"), dpi=300)
        plt.close()




# n =n *4
# for exp_name in exp:
#     try:
#         topic_diversity = load_topic_diversity(exp_name)
#         topic_diversity = np.convolve(topic_diversity, np.ones((n,)) / n, mode='valid')
#         time = load_time(exp_name)
#         time = np.convolve(time, np.ones((n,)) / n, mode='valid')
#
#
#         plt.plot(time, topic_diversity, linewidth=1)
#     except FileNotFoundError:
#         print(f"Skipping training results for {exp_name}. ")
#
#
# plt.xlabel(r"Time (s)")
# plt.ylabel("topic_diversity")
# if prefix == 'ProdLDA_20NewsGroup_2e-3target_density_':
#     topic_diversity_lim = (0.7, 0.95)
#     plt.axhline(y=0.8, color='red', linestyle='--')
# if prefix == 'ProdLDA_DBLP_2e-3target_density_':
#     topic_diversity_lim = (0.4, 0.6)
#     plt.axhline(y=0.6, color='red', linestyle='--')
#
#
#
# plt.xlim(time_lim)
# plt.ylim(topic_diversity_lim)
#
# plt.grid(linestyle="--", color='black', lw='0.5', alpha=0.5)
#
# plt.legend(den,
#            frameon=False, loc="center right")
#
# plt.savefig(join(fig_path, prefix+"topic_diversity"+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), dpi=300)
# plt.close()
#
# for exp_name in exp:
#     try:
#         npmi = load_npmi(exp_name)
#         npmi = np.convolve(npmi, np.ones((n,)) / n, mode='valid')
#         time = load_time(exp_name)
#         time = np.convolve(time, np.ones((n,)) / n, mode='valid')
#
#
#         plt.plot(time, npmi, linewidth=1)
#     except FileNotFoundError:
#         print(f"Skipping training results for {exp_name}. ")
#
#
#
#
# plt.xlabel(r"Time (s)")
# plt.ylabel("NPMI Coherence")
# if prefix == 'ProdLDA_20NewsGroup_2e-3target_density_':
#     topic_coher_lim = (-0.05, 0.13)
# if prefix == 'ProdLDA_DBLP_2e-3target_density_':
#     topic_coher_lim = (-0.05, 0.08)
#
# plt.xlim(time_lim)
# plt.ylim(topic_coher_lim)
#
# plt.grid(linestyle="--", color='black', lw='0.5', alpha=0.5)
#
# plt.legend(den,
#            frameon=False, loc="center right")
#
# plt.savefig(join(fig_path, prefix+"NPMI"+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), dpi=300)
# plt.close()
#
# for exp_name in exp:
#     try:
#         model_size = load_model_size(exp_name)
#         time = load_time(exp_name)
#
#
#         plt.plot(time, model_size, linewidth=1)
#     except FileNotFoundError:
#         print(f"Skipping training results for {exp_name}. ")
#
#
#
# plt.xlabel(r"Time (s)")
# plt.ylabel("Model Size")
#
# model_size_lim = (0, 200000)
# plt.xlim(time_lim)
# plt.ylim(model_size_lim)
#
# plt.grid(linestyle="--", color='black', lw='0.5', alpha=0.5)
#
# plt.legend(den,
#            frameon=False, loc="center right")
#
# plt.savefig(join(fig_path, prefix+"model_size"+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), dpi=300)
# plt.close()
#
# n = n // 4
name1 =['1.0','0.8','0.8*4','0.6','0.6*4','0.4','0.4*4','0.2','0.2*4','0.1','0.1*4','0.01','0.01*4']
prefix1 = 'ProdLDA_20NewsGroup_2e-3target_density_'
#prefix ='ProdLDA_M10_2e-3target_density_'
#prefix = 'ProdLDA_BBC_News_2e-3target_density_'
prefix2 = 'ProdLDA_DBLP_2e-3target_density_'
best_acc={}
best_NPMI={}
best_td={}
min_modelsize = {}
end_best_acc={}
end_best_NPMI={}
end_best_td={}
acc_time = {}
npmi_time = {}
td_time = {}
train_time = {}
all_time = {}
for name in [name1]:
    for prefix in [prefix1,prefix2]:

        #to initialize the exp
        exp = []
        for i in name:
            exp.append(prefix+i)
        # to initialize the legend
        den=[]
        for i in exp:
            if i[len(i)-2:] == '*4':
                if i[len(i)-6:-2] == '0.01':
                    den.append('Fast_'+i[len(i)-6:-2])
                else:
                    den.append('Fast_'+i[len(i)-5:-2])
            else:
                if i[len(i) - 4:] == '0.01':
                    den.append('Normal_'+i[len(i) - 4:])
                else:
                    den.append('Normal_'+i[len(i) - 3:])


        if prefix == 'ProdLDA_20NewsGroup_2e-3target_density_':
            acc_target = [0.40, 0.42, 0.43]

        if prefix == 'ProdLDA_DBLP_2e-3target_density_':
            acc_target = [0.62, 0.64, 0.645]


        for exp_name in exp:
            acc = load_acc(exp_name)
            size = load_model_size(exp_name)
            acc = np.convolve(acc, np.ones((n,)) / n, mode='valid')
            NPMI = load_npmi(exp_name)
            NPMI = np.convolve(NPMI, np.ones((n,)) / n, mode='valid')
            time = load_time(exp_name)
            min_modelsize[exp_name] = np.min(size)


            td = load_topic_diversity(exp_name)
            td = np.convolve(td, np.ones((n,)) / n, mode='valid')

            if prefix == 'ProdLDA_20NewsGroup_2e-3target_density_':
                train_time[exp_name] = (time[-2]-time[-1])/5
                all_time[exp_name] = time[-1]
                statu = True
                if exp[len(exp) - 2:] == '*4':
                    begin = 125
                else:
                     begin = 455


            if prefix == 'ProdLDA_DBLP_2e-3target_density_':
                statu = False
                all_time[exp_name] = time[-1]
                train_time[exp_name] = (time[-2] - time[-1])
                if exp[len(exp) - 2:] == '*4':
                    begin = 110
                else:
                     begin = 370

            time = np.convolve(time, np.ones((n,)) / n, mode='valid')
            best_acc[exp_name]=np.max(acc[begin:])
            best_NPMI[exp_name]=np.max(NPMI[begin:])
            best_td[exp_name] = np.max(td[begin:])
            end_best_acc[exp_name]=np.max(acc)

            max = 0

            for i in range(0,len(acc)):

                if max - acc[i] >0.03 and statu == False:
                    end_best_acc[exp_name] = np.max(acc[i:])
                    statu = True

                if max < acc[i]:
                    max = acc[i]

                if statu == True:

                    for j in acc_target:
                        key = exp_name+'acc'+str(j)
                        if key not in acc_time:
                            if acc[i]>=j:
                                acc_time[key] = time[i]






    print(best_acc,best_NPMI,best_td)



    print(end_best_acc)
    print(all_time)
    print(acc_time)



