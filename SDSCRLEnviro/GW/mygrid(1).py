
# @TIME   : 2020/4/17 5:24 PM
# @Author : ChenLiuyin
# @File   : mygrid.py
from GridWorldEnvironment import GridWorldEnv
import numpy as np
import random
env = GridWorldEnv(n_width=6,          # 水平方向格子数量
                   n_height=6,        # 垂直方向格子数量
                   n_direction=4,
                   u_size=60,         # 可以根据喜好调整大小
                   default_reward=-1, # 默认格子的即时奖励值
                   default_type=0)    # 默认的格子都是可以进入的
from gym import spaces                  # 导入spaces
env.action_space = spaces.Discrete(2)   # 设置行为空间支持的行为数量
start_x = np.random.randint(0, 5)
start_y = np.random.randint(0, 5)
start_z = np.random.randint(0, 3)
env.start = (start_x, start_y)
env.start_direction = start_z

env.refresh_setting()
env.reset()
env.render()
nfs = env.observation_space
nfa = env.action_space
print("start: %s, %s, direction: %s"%(start_x, start_y, start_z))
#print("nfs:%s; nfa:%s" % (nfs, nfa))
#print(env.observation_space)
#print(env.action_space)
#print(env.state)
n=20000
trainH=np.zeros((n,30))
trainT=np.zeros((n,1))
for step in range(n):
    # input("press any key to continue...")
    h_list = []
    for i in range(7):

        #a = env.action_space.sample()
        tou = random.random()
        if tou > 0.5:
            a = 0
            h_once = [0]
            print("act Forward")
        else:
            a = 1
            h_once = [1]
            print("act Turn Left")

        state, direc, reward, isdone, info, observation, suc = env.step(a)
        if suc:
            print("success")
        else:
            print("fail")
        print(info)
        if observation == 0:
            print("observe white")
            h_once.extend([0, 0, 0, 0, 1])
        elif observation == 1:
            print("observe red")
            h_once.extend([0, 0, 0, 1, 0])
        elif observation == 2:
            print("observe orange")
            h_once.extend([0, 0, 1, 0, 0])
        elif observation == 3:
            print("observe blue")
            h_once.extend([0, 1, 0, 0, 0])
        elif observation == 4:
            print("observe yellow")
            h_once.extend([1, 0, 0, 0, 0])
        h_list.append(h_once)
    h=np.array(h_list).flatten()
    print(np.array(h_list).flatten())
    trainH[step,:30]=h[:30]
    o1=np.argmax(h[31:31+5])
    o2=np.argmax(h[31+5+1:])
    trainT[step, 0]=o1*5+o2
    env.render()
quit()

import pickle
with open(r'dataTWTrain','wb') as f:
    pickle.dump((trainH,trainT),f)












