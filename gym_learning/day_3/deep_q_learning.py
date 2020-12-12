
"""
Agent learns the policy based on Q-learning with Deep Q-Network.
Based on the example here: https://morvanzhou.github.io/tutorials/machine-learning/torch/4-05-DQN/
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt

# Cheating mode speeds up the training process
CHEAT = True


# ----step 1 建立 Network
class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()

        # 輸入層 (state) 到隱藏層，隱藏層到輸出層 (action)   =>  use nn.Linear make sure n_states (4) to n_hidden (50) to n_actions(2)
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)
        

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x) # ReLU activation
        actions_value = self.out(x)
        return actions_value

# ----step 2 建立 Deep Q-Network
class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity):

        self.eval_net, self.target_net = Net(n_states, n_actions, n_hidden), Net(n_states, n_actions, n_hidden)
        #np.zeros(2000,10)
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2)) # 每個 memory 中的 experience 大小為 (state + next state + reward + action)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr) #get the torch.optim.adam default value
       
        self.loss_func = nn.MSELoss() #jsut get the function       !!!!!!!!!!!!!!!!!!!!!!

        self.memory_counter = 0
        self.learn_step_counter = 0 # 讓 target network 知道什麼時候要更新

        self.n_states = n_states #4
        self.n_actions = n_actions #2
        self.n_hidden = n_hidden #50
        self.batch_size = batch_size #32
        self.lr = lr #0.01
        self.epsilon = epsilon #0.1
        self.gamma = gamma #0.9
        self.target_replace_iter = target_replace_iter #100 target network 更新間隔
        self.memory_capacity = memory_capacity #2000

    #epsilon 的機率 agent 的action
    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state), 0) # torch.floattensor fucntion use for change format
        x = torch.unsqueeze(torch.FloatTensor(state), 0) # torch.floattensor fucntion use for change format || unsqueeze add the array to the number array
        # print(x)
        # epsilon-greedy
        if np.random.uniform() < self.epsilon: # 隨機
            action = np.random.randint(0, self.n_actions)

            # print('隨機')
        else: # 根據現有 policy 做最好的選擇
            actions_value = self.eval_net(x) # 以現有 eval net 得出各個 action 的分數  !!!!!!!!!!!!!!!!!!!!!!
            action = torch.max(actions_value, 1)[1].data.numpy()[0] # 挑選最高分的 action
            # print('policy')
        return action


    def store_transition(self, state, action, reward, next_state):
        # 打包 experience
        transition = np.hstack((state, [action, reward], next_state))

        # 存進 memory；舊 memory 可能會被覆蓋 || 不停存１０個的data to memory 的index
        index = self.memory_counter % self.memory_capacity  # 0 % 2000
        self.memory[index] = transition
        self.memory_counter += 1


    def learn(self):
        # 隨機取樣 batch_size 32 個 experience
        sample_index = np.random.choice(self.memory_capacity, self.batch_size) # np.random.choice( the range is memory_capacity,32 numbers)
        b_memory = self.memory[sample_index, :]

        #主要拿回2000表中抽的32個array中的各個值
        b_state = torch.FloatTensor(b_memory[:, :self.n_states])
        b_action = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int))
        b_reward = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])
        b_next_state = torch.FloatTensor(b_memory[:, -self.n_states:])

        # 計算現有 eval net 和 target net 得出 Q value 的落差
        q_eval = self.eval_net(b_state).gather(1, b_action) # 重新計算這些 experience 當下 eval net 所得出的 Q value　｜｜　在eval_net result 中抽出　b_action 位置的值

        q_next = self.target_net(b_next_state).detach() # detach 才不會訓練到 target net || 將下個　next state 計完的result => detch 一般都是用来计算一些其他的辅助变量，用以debug，这是比较多

        q_target = b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1) # 計算這些 experience 當下 target net 所得出的 Q value

        loss = self.loss_func(q_eval, q_target) # 根據q_eval和q_target的影響計算出loss的值　　
        

        # Backpropagation 梯度下降法
        self.optimizer.zero_grad()#梯度初始化为零

        loss.backward() #即反向传播求梯度

        self.optimizer.step()#即更新所有参数

        # 每隔一段時間 (target_replace_iter), 更新 target net，即複製 eval net 到 target net
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

env = gym.make('CartPole-v0')

# Environment parameters
n_actions = env.action_space.n # environment actions numbers => 2
n_states = env.observation_space.shape[0]#observation [小車位置，小車速度(負左，正右)，柱子角度(向左負，向右正)，柱尖速度] => 4

# Hyper parameters
n_hidden = 50             #the nerual network hidden size

batch_size = 32           #get the batch size for the matrix
lr = 0.01                 # learning rate
epsilon = 0.1            # epsilon-greedy
gamma = 0.9               # reward discount factor
target_replace_iter = 100 # target network 更新間隔
memory_capacity = 1000

n_episodes = 400

# 建立 DQN
dqn = DQN(n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity)

# 學習
all_rewards = []
for i_episode in range(n_episodes):
    t = 0
    rewards = 0
    
    state = env.reset()
    # print('base:',state)
    print(i_episode)
    while True:
        env.render()

        # 選擇 action
        action = dqn.choose_action(state)
        next_state, reward, done, info = env.step(action)


        # Cheating part: modify the reward to speed up training process 
        # 建立 reward 分配方法 => 小車保持在中間，那麼小車跟中間的距離越小，reward 也應該越大
        if CHEAT:
            x, v, theta, omega = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8 # reward 1: the closer the cart is to the center, the better
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5 # reward 2: the closer the pole is to the center, the better
            reward = r1 + r2

        # 儲存 experience
        dqn.store_transition(state, action, reward, next_state)

        # 累積 reward
        rewards += reward

        # 有足夠 experience 後進行訓練
        if dqn.memory_counter > memory_capacity:
            dqn.learn()

        # 進入下一 state
        state = next_state

        if done:
            print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
            all_rewards.append(t+1)
            break
        t += 1

plt.plot(all_rewards)
plt.ylabel('rewards:')
plt.xlabel('time steps:')
plt.show()

env.close()