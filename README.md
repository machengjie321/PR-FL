# Description
This is the official code for "Federated Learning based on Pruning and Recovery"
It can be seen in https://arxiv.org/abs/2403.15439,


This framework integrates asynchronous learning algorithms and pruning techniques, effectively addressing the inefficiencies of traditional federated learning algorithms in scenarios involving heterogeneous devices, as well as tackling the staleness issue and inadequate training of certain clients in asynchronous algorithms. Through the incremental restoration of model size during training, the framework expedites model training while preserving model accuracy. Furthermore, enhancements to the federated learning aggregation process are introduced, incorporating a buffering mechanism to enable asynchronous federated learning to operate akin to synchronous learning. Additionally, optimizations in the process of the server transmitting the global model to clients reduce communication overhead. Our experiments across various datasets demonstrate that: (i) significant reductions in training time and improvements in convergence accuracy are achieved compared to conventional asynchronous FL and HeteroFL; (ii) the advantages of our approach are more pronounced in scenarios with heterogeneous clients and non-IID client data.

The model pruning framework is based on Prune_Fl, https://github.com/jiangyuang/PruneFL.

# Slides of my work

![image](https://github.com/machengjie321/PR-FL/blob/main/img/background1.PNG)
![image](https://github.com/machengjie321/PR-FL/blob/main/img/background2.PNG)
![image](https://github.com/machengjie321/PR-FL/blob/main/img/my_work1.PNG)
![image](https://github.com/machengjie321/PR-FL/blob/main/img/my_work2.PNG)
![image](https://github.com/machengjie321/PR-FL/blob/main/img/my_work3.PNG)
![image](https://github.com/machengjie321/PR-FL/blob/main/img/my_work4.PNG)
![image](https://github.com/machengjie321/PR-FL/blob/main/img/mw5.PNG)
![image](https://github.com/machengjie321/PR-FL/blob/main/img/mw6.PNG)
![image](https://github.com/machengjie321/PR-FL/blob/main/img/mw7.PNG)
![image](https://github.com/machengjie321/PR-FL/blob/main/img/mw8.PNG)
![image](https://github.com/machengjie321/PR-FL/blob/main/img/mw9.PNG)
![image](https://github.com/machengjie321/PR-FL/blob/main/img/mw10.PNG)
![image](https://github.com/machengjie321/PR-FL/blob/main/img/mw11.PNG)
![image](https://github.com/machengjie321/PR-FL/blob/main/img/mw12.PNG)

# Setup
## Initial the Pruning Library file from Prune_FL
```
sudo -E python3 setup.py install
```
## Setup the environment path
```
source setenv.sh     #setup the path
```
# Download the datasets
such as Cifar10, Femnist, Tiny-ImageNet, Mnist

the details can be seen in bases/vision/datasets. 

Some datasets can be download directly, others need to be manually downloaded and placed in a specified folder.


# Run the experiments
you can see autorun/experiment.sh
It has some experimental examples from my prior work

```
conda activate /data/mcj/conda_env/d2l  #activate the conda environment
cd /data/mcj/Prune_fl # cd the work space
source setenv.sh     #setup the environment path

# Description of parameters to be added
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m fed_avg -wd 0 -md 0.08 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

```
- -ic: model density increment, such as 0.2 is the model density add 0.2 when client model need to recover
- -i: adjust interval, such as 50 is adjust the model every 50 rounds
- -f: This was an attempt in the past where I wanted to offset the effects of model pruning by adjusting the number of client iterations, you do not need to change it. 
- -d: This is a past attempt to adjust the coefficient of change in the number of iterations of the client with the pruning rate, you do not need to change it.
- -g: select the gpu id
- -m: the function of server merge the client model. such as fedavg, maskfedavg, R2SP, buffmaskfedavg
- -wd: I also tried to test the effect of weight decay, you do not need to change it. 
- -md: The initial minimum model for the client
- -ac: the evaluate metric for model pruning, you do not to change it
- -wdn: I also tried to attempt to offset the effects of model pruning by adjusting the learning rate, you do not to change it.
- -ft: I also tried to attempt to offset the effects of model pruning by adding the finetune, you do not to change it.
- -ch: asyn or syn
- -stal: the function of staleness from fedAsyn, you do not to change it 
- -re: if have it, the model will recover in the late stage of model training
- -niid: if have it, the client data is non-iid, otherwise the clients data is iid.
- -Res: if have it, use Residual split


# Analysis

You can see some of my visualizations in the dataset.jupyter


