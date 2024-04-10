import torch
import os
num_users = 199
list_users = [[i] for i in range(num_users)]

def get_indices_list():
    cur_pointer = 0
    indices_list = []
    for ul in list_users:
        num_data = 0
        for user_id in ul:
            train_meta = torch.load(os.path.join("..","..","datasets", "FEMNIST", "processed", "train_{}.pt".format(user_id)))
            num_data += len(train_meta[0])
        indices_list.append(list(range(cur_pointer, cur_pointer + num_data)))
        cur_pointer += num_data

    return indices_list

indices_list = get_indices_list()
from sample import client_data
sequence = client_data(indices_partition=indices_list,num_round=500,data_per_client=50,client_selection=False,client_per_round=10)

print(sequence)