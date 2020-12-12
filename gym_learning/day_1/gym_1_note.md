## gym Q-learning 學習筆記

>> install gym
"""linux
	pip install gym
"""

>> step 1
first userandom action to try the game and get the rewards ,
we can see the rewards area in 20-30

>> step 2 
try add the policy to control the action 
we find that the rewards area in 30 - 50

>> step 3
main use the two function [choose_action() ,get_state() ]
>> set the value function:
"""python
	get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/25)))  # epsilon-greedy; 隨時間遞減
	get_lr = lambda i: max(0.01, min(0.5, 1.0 - math.log10((i+1)/25))) # learning rate; 隨時間遞減 
	gamma = 0.99 # reward discount factor
"""
>> main formula
"""python
    q_next_max = np.amax(q_table[next_state]) # 進入下一個 state 後，預期得到最大總 reward
    q_table[state + (action,)] += lr * (reward + gamma * q_next_max - q_table[state + (action,)]) # 就是那個公式
"""

# resource
https://medium.com/pyladies-taiwan/reinforcement-learning-%E5%81%A5%E8%BA%AB%E6%88%BF-openai-gym-e2ad99311efc
