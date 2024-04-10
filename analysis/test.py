# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import lognorm
#
# mean_values = [0]  # 不同均值值
# variance = [0.1,0.3,0.5,0.8,1,2,3]  # 方差
#
# # 生成 x 值范围
# x = np.linspace(0.01, 5, 1000)
#
# # 绘制不同均值的对数正态分布曲线
# for mean in mean_values:
#     for var in variance:
#         # 计算对应均值和方差的对数正态分布曲线
#         y = lognorm.pdf(x, s=np.sqrt(var), loc=0, scale=np.exp(mean))
#
#         # 绘制分布曲线
#         plt.plot(x, y, label=f'Mean = {mean}.Var = {var}')
#
# # 设置图例和标题
# plt.legend()
# plt.title('Lognormal Distribution')
# plt.xlabel('x')
# plt.ylabel('Probability Density')
# plt.show()

import heapq


def simulate_client_to_server(standard_time, time, size, upload_speed, server_download_speed):
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

    for i in range(n):
        heapq.heappush(min_time_heap, time[i])
        heapq.heappush(min_time_heap, time[i] + size[i] / upload_speed[i])

    next_time = heapq.heappop(min_time_heap)
    current_time = next_time
    # 存储每个客户端上传到网络的文件大小
    uploaded_size = [0] * n

    next_time = heapq.heappop(min_time_heap)

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
            if state[i] == 1:
                cycle_size[i] = min(upload_speed[i] * (next_time - current_time), size[i])
                uploaded_size[i] += cycle_size[i]
                size[i] -= cycle_size[i]
                if size[i] == 0:
                    state[i] = 2

        # 模拟服务器下载文件的过程

        max_download_size = server_download_speed * (next_time - current_time)

        for j in range(n):
            if max_download_size >= uploaded_size[j]:
                max_download_size -= uploaded_size[j]
                uploaded_size[j] = 0
                if state[j] == 2:
                    state[j] = 3
                    server_receive_time[j] = standard_time + next_time

                    print(str(standard_time + next_time) + "s: The data of client" + str(
                        j) + " has been  uploaded successfully  ")

            else:
                uploaded_size[j] -= max_download_size

                break

        current_time = next_time
        if len(min_time_heap) > 0:
            next_time = heapq.heappop(min_time_heap)
        else:

            next_time = current_time + (sum(uploaded_size) + sum(size)) / server_download_speed

        if sum(state) == n * 3:
            break
        else:
            print('continue')

    return server_receive_time
standard_time = 0
time =[0,1,2,3,4,5,6,7,8,9]
size=[25.193674087524414, 25.193674087524414, 25.193674087524414, 25.193674087524414, 25.193674087524414, 25.193674087524414, 25.193674087524414, 25.193674087524414, 25.193674087524414, 25.193674087524414]
upload_speeds=[3.8645008119347746, 5.988968710612888, 4.143450862689236, 1.88220259971606, 0.750168614549234, 1.573619171957287, 1.0474053308441704, 0.6991465401158139, 0.5372047667136786, 0.38897161024712507]
server_download_speed = 5

simulate_client_to_server(standard_time, time, size, upload_speeds, server_download_speed)


