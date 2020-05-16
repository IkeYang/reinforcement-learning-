#Author:ike yang
import numpy as np
import matplotlib.pyplot as plt
import copy

def gridWorldStep(s,a):
    #s(x,y)
    #a 1,2,3,4
    s=copy.copy(s)
    if a==1:
        s[0]+=1
    elif a==2:
        s[1] += 1
    elif a==3:
        s[0] -= 1
    elif a==4:
        s[1] -= 1

    if s[1]==0:
        if s[0]==0:
            return s, -1
        elif s[0]==11:
            return s,0
        else:
            return [0,0],-100
    else:
        return s, -1

def getActionSpace(s):
    x=s[0]
    y=s[1]
    if x==0 and y==0:
        return [1,2]
    if x==0 and y<3 and y>0:
        return [1,2,4]
    if x==0 and y==3:
        return [1,4]
    if x>0 and x<11 and y==3:
        return [1,3,4]
    if x == 11 and y == 3:
        return [3, 4]
    if x == 11 and y<3:
        return [2, 3, 4]
    return [1,2, 3, 4]

def initQ():
    #Q[x][y][a]=v
    Q={}
    for x in range(12):
        Q[x]={}
        for y in range(4):
            if y==0 and x<11 and x>0:
                continue
            if y==0:
                pass
            Q[x][y]={}
            aL=getActionSpace([x,y])
            for a in aL:
                Q[x][y][a]=0

    return Q


def e_GreedychooseMaxAction(Q,s,e):
    x=s[0]
    y=s[1]
    aL=getActionSpace([x,y])
    maxAv=-float('inf')

    for i,a in enumerate(aL):
        av=Q[x][y][a]
        if av>maxAv:
            maxA=a
            maxAv=av
            ind=i
    p=np.zeros((len(aL)))
    p+=e/len(aL)
    p[ind]+=1-e
    return np.random.choice(aL, p=p)



alpha=0.1
e=0.1
n=2000

#sarsa
Q=initQ()
stop=False
sumReward = 0
sumRewardL=[]
for p in range(n):
    # e=0.1/(p+1)
    s0 = [0, 0]
    # sumReward = 0
    a=e_GreedychooseMaxAction(Q,s0,e)
    while not stop:
        s1,r=gridWorldStep(s0, a)
        sumReward+=r
        a2=e_GreedychooseMaxAction(Q,s1,e)
        Q[s0[0]][s0[1]][a] += alpha*(r+Q[s1[0]][s1[1]][a2]-Q[s0[0]][s0[1]][a])
        s0=s1
        a=a2
        if s0[0]==11 and s0[1]==0:
            break
    sumRewardL.append(sumReward/(p+1))




#Qlearning
Q=initQ()
stop=False
sumReward = 0
sumRewardL2=[]
for p in range(n):
    # e=0.1/(p+1)
    s0 = [0, 0]
    # sumReward = 0
    while not stop:
        a = e_GreedychooseMaxAction(Q, s0, e)
        s1,r=gridWorldStep(s0, a)
        sumReward+=r
        a2=e_GreedychooseMaxAction(Q,s1,0)
        Q[s0[0]][s0[1]][a] += alpha*(r+Q[s1[0]][s1[1]][a2]-Q[s0[0]][s0[1]][a])
        s0=s1
        if s0[0]==11 and s0[1]==0:
            break
    sumRewardL2.append(sumReward/(p+1))
plt.plot(sumRewardL, label='Sarsa')
plt.plot(sumRewardL2,label='Q-Learning')
plt.legend()
plt.show()
plt.plot(sumRewardL[800:], label='Detail after 800 iters: Sarsa')
plt.plot(sumRewardL2[800:],label='Detail after 800 iters: Q-Learning')
plt.legend()
plt.show()



















