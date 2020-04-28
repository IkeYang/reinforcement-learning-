#Author:ike yang
import numpy as np

import time

def UpdateTunnelWorld(s0,p):
    '''

    :param s0: last s
    :param p:
    :return: s1,ob1,action
    '''
    rd = np.random.rand()
    if s0== 1:
        if rd < p:
            return 1,1,0
        else:
            return 2,0,1
    if s0== 2:
        if rd < p:
            return 1,1,-1
        else:
            return 3,0,1
    if s0== 3:
        if rd < p:
            return 2,0,-1
        else:
            return 4,0,1
    if s0== 4:
        if rd < p:
            return 3,0,-1
        else:
            return 4,0,1
def SimulateTunnelWorld(step=50,state=None,p=0.7,verbose=False):
    '''

    :param step: Max step
    :param state: init state
    :param p: transition Probablity
    :return: S a list of state {1,2,3,4} with 50 len
            Ob observation a list of observation {0: dark, 1: light} with 50 len
            action[-1,0,1] with 50 len
    '''
    S=np.zeros((step,))
    Ob=np.zeros((step,))
    ac=np.zeros((step,))
    if state is None:
        S[0]=1
        Ob[0]=1
        ac[0]=0
        for i in range(1, step):
            S[i], Ob[i], ac[i] = UpdateTunnelWorld(S[i - 1], p)
            if verbose:
                time.sleep(0.2)
                print("\r State " + str(S[:i]), end="")
                # print("\r Observation ",Ob[:i], end="")
    else:
        S[0] = state
        _, Ob[0], ac[0] = UpdateTunnelWorld(S[0], p)
    for i in range(1,step):
        S[i],Ob[i], ac[i]=UpdateTunnelWorld(S[i-1], p)
        if verbose:
            time.sleep(0.2)
            print("\r State "+str(S[:i]), end="")
            # print("\r Observation ",Ob[:i], end="")
    return S,Ob,ac
def UpdateHalfMoonWorld(s0):
    '''

    :param s0: 0-19
    :return:
    '''
    rd = np.random.rand()
    if rd<0.5:
        s=(s0+1)%20
        if s<10:
            return s,1
        else:
            return s, 0
    else:
        s = (s0 - 1+20) % 20
        if s < 10:
            return s, 1
        else:
            return s, 0
def SimulateHalfMoonWorld(step=50,state=None,verbose=False):
    '''

    :param step: Max step
    :param state: init state
    :param p: transition Probablity
    :return: S a list of state {0-19} with 50 len
            Ob observation a list of observation {0: dark, 1: light} with 50 len
    '''
    S=np.zeros((step,))
    Ob=np.zeros((step,))
    if state is None:
        S[0]=0
        Ob[0]=1
    else:
        S[0] = state
    for i in range(1,step):
        S[i],Ob[i]=UpdateHalfMoonWorld(S[i-1])
        if verbose:
            time.sleep(0.2)
            print("\r State "+str(S[:i]), end="")
            # print("\r Observation ",Ob[:i], end="")
    return S,Ob



# s,ob,ac=SimulateTunnelWorld(step=500)
# print(s)
# print(ob)
# print(ac)
# s,ob=SimulateHalfMoonWorld(step=500)
# print(s)
# print(ob)


def makeData(n,stpe=5):
    #h=5 t=2
    H=np.zeros((n,2*stpe))
    T=np.zeros((n,1))
    state=None
    for i in range(n):
        s, ob, ac = SimulateTunnelWorld(state=state,step=stpe+2)
        state=s[-1]
        oa=np.hstack((ob.reshape((-1,1)),ac.reshape((-1,1)))).flatten()
        H[i,:]=oa[:stpe*2]
        t1=oa[stpe*2]
        t2=oa[stpe*2+2]
        T[i,0]=t2*2+t1

    return H,T

trainH,trainT=makeData(20000,10)
import pickle
with open(r'C:\Users\ylx\OneDrive(2).old\code\RL\RLCourse\dataTW10Train','wb') as f:
    pickle.dump((trainH,trainT),f)


trainH,trainT=makeData(5000,10)
import pickle
with open(r'C:\Users\ylx\OneDrive(2).old\code\RL\RLCourse\dataTW10Val','wb') as f:
    pickle.dump((trainH,trainT),f)
