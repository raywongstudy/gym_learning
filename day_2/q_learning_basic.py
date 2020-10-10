import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#设置Gamma参数和环境奖励R
GAMMA = 0.8
R=np.asarray([[-1,-1,-1,-1,0,-1],
   [-1,-1,-1,0,-1,100],
   [-1,-1,-1,0,-1,-1],
   [-1,0, 0, -1,0,-1],
   [0,-1,-1,0,-1,100],
   [-1,0,-1,-1,0,100]])
#初始化Q
Q = np.zeros((6,6))
#寻找最大的奖励
def getMaxQ(state):
    return max(Q[state, :])
#Q-Learning
def QLearning(state):
    curAction = None
    #while 
    for action in range(6):
        if(R[state][action] == -1):
                Q[state, action]=0
        else:
            curAction = action
            Q[state,action]=R[state][action]+GAMMA * getMaxQ(curAction) #公式

#主函数 進行200次不停更新和令數計最優
count=0
while count<200:
    for i in range(6):
        QLearning(i)
    count+=1

# sns 用作出圖片
sns.set()
f, ax = plt.subplots(figsize=(8, 6))
cmap = sns.diverging_palette(230, 10)
sns.heatmap(Q, cmap = cmap, annot=True, fmt='g', linewidths=.5, ax=ax)
print(Q) #final q table
plt.show() # show the q table graph 