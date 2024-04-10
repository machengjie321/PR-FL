import numpy as np

# 定义初始状态空间大小和动作空间大小
initial_num_states = 101  # 初始状态空间大小
num_actions = 2

# 初始化Q值函数和状态空间的左边界
Q = np.zeros((initial_num_states, num_actions))
left_boundary = 0  # 初始状态空间的左边界

# 定义Q-learning算法函数
def q_learning(num_episodes, learning_rate, discount_factor, exploration_prob):
    global left_boundary  # 使用全局变量来跟踪状态空间的左边界
    
    for episode in range(num_episodes):
        state = np.random.randint(left_boundary, initial_num_states)  # 随机选择一个初始状态
        
        while state != initial_num_states - 1:  # 终止状态为最大状态值
            # 选择动作，这里使用ε-greedy策略
            if np.random.rand() < exploration_prob:
                action = np.random.randint(num_actions)  # 随机动作
            else:
                action = np.argmax(Q[state, :])  # 最优动作
            
            next_state = state + action  # 状态转移
            reward = 1 if next_state == initial_num_states - 1 else 0  # 终止状态有奖励，其他状态无奖励
            
            # Q值更新，使用Q-learning更新公式
            Q[state, action] += learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])
            
            state = next_state
        
        # 每训练10轮，更新状态空间的左边界
        if episode % 10 == 0 and left_boundary < initial_num_states - 1:
            left_boundary += 1
    
    return Q

# 设置参数并运行Q-learning算法
num_episodes = 1000
learning_rate = 0.1
discount_factor = 0.9
exploration_prob = 0.1

final_Q = q_learning(num_episodes, learning_rate, discount_factor, exploration_prob)
print("Final Q-values:")
print(final_Q)
