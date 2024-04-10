from time import time
import multiprocessing as mp

import torchvision
from torchvision import transforms

import torch
import os
from configs.celeba import *
import configs.celeba as config
from bases.vision.load import get_data_loader

if os.getcwd().startswith('/mnt/sda1/mcj/PruneFL-master/PruneFL-master'):
    os.chdir('/mnt/sda1/mcj/PruneFL-master/PruneFL-master')

if os.getcwd().startswith("/data/mcj/Prune_fl"):
    os.chdir("/data/mcj/Prune_fl")
transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(
    root='dataset/',
    train=True,  # 如果为True，从 training.pt 创建数据，否则从 test.pt 创建数据。
    download=True,  # 如果为true，则从 Internet 下载数据集并将其放在根目录中。 如果已下载数据集，则不会再次下载。
    transform=transform
)
from torch.utils.data import DataLoader, SubsetRandomSampler

# train_loader = get_data_loader(EXP_NAME, data_type="val", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
#                                num_workers=3, pin_memory=True)
# # 计算要抽样的子集大小（假设是原数据集大小的 1/10）
# subset_size = len(train_loader) // 10
# # 生成随机的子集索引
# indices = torch.randperm(len(train_loader))[:subset_size]
# # 使用 SubsetRandomSampler 创建新的数据加载器
# subset_sampler = SubsetRandomSampler(indices)
# subset_data_loader = DataLoader(dataset=train_loader.dataset, batch_size=train_loader.batch_size,
#                                 sampler=subset_sampler)
import torch

# 设置默认的 GPU 设备为第一个 GPU
torch.cuda.set_device(1)
print(f"num of CPU: {mp.cpu_count()}")
for num_workers in range(0, 50, 1):
    # subset_data_loader = DataLoader(dataset=train_loader.dataset, batch_size=train_loader.batch_size,
    #                                 sampler=subset_sampler,pin_memory=True,num_workers=num_workers)
    train_loader = get_data_loader(EXP_NAME, data_type="test", num_workers=8, batch_size=100, shuffle=False,
                                           pin_memory=True)

    start = time()
    for epoch in range(1, 2):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

    '''num of CPU: 64
num of CPU: 64
Data already downloaded.
Finish with:14.418375253677368 second, num_workers=8
Data already downloaded.
Finish with:10.657771110534668 second, num_workers=9
Data already downloaded.
Finish with:9.931421279907227 second, num_workers=10
Data already downloaded.
Finish with:9.335034132003784 second, num_workers=11
Data already downloaded.
Finish with:8.421121835708618 second, num_workers=12
Data already downloaded.
Finish with:8.219359874725342 second, num_workers=13
Data already downloaded.
Finish with:8.102792024612427 second, num_workers=14
Data already downloaded.
Finish with:7.386909246444702 second, num_workers=15
Data already downloaded.
Finish with:6.862617254257202 second, num_workers=16
Data already downloaded.
Finish with:7.230368614196777 second, num_workers=17
Data already downloaded.
Finish with:6.928982973098755 second, num_workers=18
Data already downloaded.
Finish with:6.842309951782227 second, num_workers=19
Data already downloaded.
Finish with:6.9146294593811035 second, num_workers=20
Data already downloaded.
Finish with:6.633349895477295 second, num_workers=21
Data already downloaded.
Finish with:6.718991994857788 second, num_workers=22
Data already downloaded.
Finish with:6.613035440444946 second, num_workers=23
(/mnt/sda1/mcj/conda_env/d2l) chengjie@omnisky:/mnt/sda1/mcj/PruneFL-master/PruneFL-master$ python main.py 
num of CPU: 64
Data already downloaded.
Finish with:8.706851243972778 second, num_workers=24
Data already downloaded.
^Z
[2]+  Stopped                 python main.py
(/mnt/sda1/mcj/conda_env/d2l) chengjie@omnisky:/mnt/sda1/mcj/PruneFL-master/PruneFL-master$ python main.py 
num of CPU: 64
Data already downloaded.
Finish with:14.477522134780884 second, num_workers=8
Data already downloaded.
Finish with:9.852701663970947 second, num_workers=10
Data already downloaded.
Finish with:8.318182229995728 second, num_workers=12
Data already downloaded.
Finish with:7.671738386154175 second, num_workers=14
Data already downloaded.
Finish with:6.926000118255615 second, num_workers=16
Data already downloaded.
Finish with:6.878815412521362 second, num_workers=18
Data already downloaded.
Finish with:6.8193840980529785 second, num_workers=20
Data already downloaded.
Finish with:6.585986137390137 second, num_workers=22
Data already downloaded.
Finish with:7.125152587890625 second, num_workers=24
Data already downloaded.
Finish with:7.0888707637786865 second, num_workers=26
Data already downloaded.
Finish with:7.17974066734314 second, num_workers=28
Data already downloaded.
Finish with:6.817237615585327 second, num_workers=30
Data already downloaded.
Finish with:7.389444351196289 second, num_workers=32
Data already downloaded.
Finish with:7.38226842880249 second, num_workers=34
Data already downloaded.
Finish with:7.869073867797852 second, num_workers=36
Data already downloaded.
Finish with:7.711442232131958 second, num_workers=38
Data already downloaded.
Finish with:7.841994524002075 second, num_workers=40
Data already downloaded.
Finish with:8.558963298797607 second, num_workers=42
Data already downloaded.
Finish with:7.5597662925720215 second, num_workers=44
Data already downloaded.
Finish with:8.153611660003662 second, num_workers=46
Data already downloaded.
Finish with:8.339036464691162 second, num_workers=48'''