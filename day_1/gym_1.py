import gym
from time import sleep
import math
import numpy as np

import matplotlib.pyplot as plt
#用random 玩game--------------- step 1
# env = gym.make('CartPole-v1')
# # 跑 200 個 episode，每個 episode 都是一次任務嘗試
# for i_episode in range(200):
#     observation = env.reset() # 讓 environment 重回初始狀態 
#     rewards = 0 # 累計各 episode 的 reward 
#     for t in range(250): # 設個時限，每個 episode 最多跑 250 個 action
#         env.render() # 呈現 environment

#         # Key section
#         action = env.action_space.sample() # 在 environment 提供的 action 中隨機挑選
#         observation, reward, done, info = env.step(action) # 進行 action，environment 返回該 action 的 reward 及前進下個 state

#         rewards += reward # 累計 reward

#         if done: # 任務結束返回 done = True
#             print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
#             break
# env.close()
# ------------------- 結果 -- 》整體 reward 並不高。


#加入教學policy ------------------ step 2

# # 定義 policy
# def choose_action(observation):
#     pos, v, ang, rot = observation
#     return 0 if ang < 0 else 1 # 柱子左傾則小車左移，否則右移 

# action 改為下面
# action = choose_action(observation)
# ------------------ 結果 -- 》整體 reward 提高了一點點


# 定義 policy
def choose_action(state, q_table, action_space, epsilon):
    if np.random.random_sample() < epsilon: # 有 ε 的機率會選擇隨機 action
        return action_space.sample() 
    else: # 其他時間根據現有 policy 選擇 action，也就是在 Q table 裡目前 state 中，選擇擁有最大 Q value 的 action
        return np.argmax(q_table[state]) 

def get_state(observation, n_buckets, state_bounds):
	#observation [小車位置，小車速度(負左，正右)，柱子角度(向左負，向右正)，柱尖速度]
    state = [0] * len(observation) 
    for i, s in enumerate(observation): # 每個 feature 有不同的分配
        l, u = state_bounds[i][0], state_bounds[i][1] # 每個 feature 值的範圍上下限
        if s <= l: # 低於下限，分配為 0
            state[i] = 0
        elif s >= u: # 高於上限，分配為最大值
            state[i] = n_buckets[i] - 1
        else: # 範圍內，依比例分配
            state[i] = int(((s - l) / (u - l)) * n_buckets[i])

    return tuple(state)

env = gym.make('CartPole-v1')
# 準備 Q table
## Environment 中各個 feature 的 bucket 分配數量
## 1 代表任何值皆表同一 state，也就是這個 feature 其實不重要
n_buckets = (1, 1, 6, 3)

final_result = []

## Action 數量 
n_actions = env.action_space.n

## State 範圍 
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-0.5, 0.5]
state_bounds[3] = [-math.radians(50), math.radians(50)]

## Q table，每個 state-action pair 存一值 
q_table = np.zeros(n_buckets + (n_actions,))

# 一些學習過程中的參數
get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/25)))  # epsilon-greedy; 隨時間遞減
get_lr = lambda i: max(0.01, min(0.5, 1.0 - math.log10((i+1)/25))) # learning rate; 隨時間遞減 
gamma = 0.99 # reward discount factor

# Q-learning
for i_episode in range(400):
    epsilon = get_epsilon(i_episode)
    lr = get_lr(i_episode)
    print(i_episode)
    observation = env.reset()
    rewards = 0
    state = get_state(observation, n_buckets, state_bounds) # 將連續值轉成離散 
    for t in range(200):
        env.render()

        action = choose_action(state, q_table, env.action_space, epsilon)
        observation, reward, done, info = env.step(action)

        rewards += reward
        next_state = get_state(observation, n_buckets, state_bounds)

        # 更新 Q table
        q_next_max = np.amax(q_table[next_state]) # 進入下一個 state 後，預期得到最大總 reward
        q_table[state + (action,)] += lr * (reward + gamma * q_next_max - q_table[state + (action,)]) # 就是那個公式

        # 前進下一 state 
        state = next_state

        if done:
            print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
            break
    final_result.append(rewards)
env.close()

with open('model.txt', 'w') as outfile:
    for slice_2d in q_table[0][0]:
        np.savetxt(outfile, slice_2d)

plt.plot(final_result)
plt.show()

