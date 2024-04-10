# Description
This is the official code for "Federated Learning based on Pruning and Recovery"
It can be seen in https://arxiv.org/abs/2403.15439,


This framework integrates asynchronous learning algorithms and pruning techniques, effectively addressing the inefficiencies of traditional federated learning algorithms in scenarios involving heterogeneous devices, as well as tackling the staleness issue and inadequate training of certain clients in asynchronous algorithms. Through the incremental restoration of model size during training, the framework expedites model training while preserving model accuracy. Furthermore, enhancements to the federated learning aggregation process are introduced, incorporating a buffering mechanism to enable asynchronous federated learning to operate akin to synchronous learning. Additionally, optimizations in the process of the server transmitting the global model to clients reduce communication overhead. Our experiments across various datasets demonstrate that: (i) significant reductions in training time and improvements in convergence accuracy are achieved compared to conventional asynchronous FL and HeteroFL; (ii) the advantages of our approach are more pronounced in scenarios with heterogeneous clients and non-IID client data.

The model pruning framework is based on Prune_Fl, https://github.com/jiangyuang/PruneFL.

# Setup
## Initial the Pruning Library file from Prune_FL

sudo -E python3 setup.py install

## Setup the environment path

source setenv.sh     #setup the path

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


# Analysis

You can see some of my visualizations in the dataset.jupyter
