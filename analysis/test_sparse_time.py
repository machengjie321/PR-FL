import copy

from bases.nn.models.leaf import Conv2
import torch


import argparse
import torch
import configs.femnist as config
import configs.network as network_config
from bases.nn.models.leaf import Conv2
from bases.vision.load import get_data_loader
from bases.optim.optimizer import SGD
from bases.fl.modules import Server
from bases.optim.optimizer_wrapper import OptimizerWrapper
from control.algorithm import ControlModule
from control.utils import ControlScheduler
from utils.functional import disp_num_params
from timeit import default_timer as timer
import os
from bases.nn.models.leaf import Conv2
import torch
seed = 0
os.chdir("/data/mcj/PruneFL-master")
train_loader = get_data_loader(config.EXP_NAME, data_type="train",
                                    batch_size=64, shuffle=True, num_workers=0,
                                    pin_memory=True)

inputs, labels = train_loader.get_next_batch()
model = Conv2()
full_model =  model.to_sparse()
full_optimizer = SGD(full_model.parameters(), lr=config.INIT_LR)
full_optimizer_wrapper = OptimizerWrapper(full_model, full_optimizer)
full_control = ControlModule(model=full_model, config=config)
num_pre_batch = 10
full_inputs, full_outputs = [], []
for _ in range(num_pre_batch):
    inp, out = train_loader.get_next_batch()
    full_inputs.append(inp)
    full_outputs.append(out)

start = timer()
for server_i in range(1, 50 + 1):
    for server_inp, server_out in zip(full_inputs, full_outputs):
        list_grad = full_optimizer_wrapper.step(server_inp, server_out)

test_loader = get_data_loader(config.EXP_NAME, data_type="test", num_workers=0, pin_memory=True)
loss, acc = full_model.evaluate(test_loader)
print(acc, timer()-start)#0.46612401244912616 199.47704527294263

for i in range(1, 10):
    sparse_model = copy.deepcopy(model).prune_by_pct(i * 0.1).to_sparse()
    sparse_optimizer = SGD(sparse_model.parameters(), lr=config.INIT_LR)
    sparse_optimizer_wrapper = OptimizerWrapper(sparse_model, sparse_optimizer)
    sparse_control = ControlModule(model=sparse_model, config=config)
    num_pre_batch = 10
    start = timer()
    for server_i in range(1, 50 + 1):

        for server_inp, server_out in zip(full_inputs, full_outputs):
            list_grad = sparse_optimizer_wrapper.step(server_inp, server_out)


    loss, acc = sparse_model.evaluate(test_loader)
    print(i, acc, timer() - start)#0.36772803447450325 120.33273744396865

'''
0 0.4675604500837922 203.8547452511266
1 0.45343548000957623 188.38706794800237
2 0.44529566674646875 167.8755925721489
3 0.45918123054824034 153.65338517120108
4 0.459660043093129 132.9847617750056
5 0.4158486952358152 116.00620408914983
6 0.38472587981805123 87.59239238500595
7 0.3057218099114197 72.13796942774206
8 0.17141489107014604 68.120925900992
9 0.07780703854440986 51.65890554198995

'''
