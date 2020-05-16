#Author:ike yang
import numpy as np
import matplotlib.pyplot as plt
import copy



def randowWalk():
    # state 0,1,2,3,4,5,6
    s0=3
    s=[]
    s.append(s0)
    a=[]
    r=[]
    while s0>0 and s0<6:
        aa=np.random.random()
        if aa>0.5:
            s0+=1
            s.append(s0)
            a.append(1)
            if s0==6:
                r.append(1)
            else:
                r.append(0)
        else:
            s0 -= 1
            s.append(s0)
            a.append(-1)
            r.append(0)
    return s,a,r


def calcG(r,gamma=1):
    # r2=copy.copy(r)
    g=copy.copy(r)
    for i in range(len(r)):
        for j in range(i,len(r)):
            g[i]+=r[j]*(gamma**j)
    return g
# print(randowWalk())
def rms(a,b):
    a=np.array(a)
    b=np.array(b)
    return np.mean(np.sqrt((a-b)*(a-b)))


#mc
def trainMC(alpha):
    vT = [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
    value=np.ones((5))/2
    rms0=0
    rmsL=[]
    for h in range(5000):
        s,_,r=randowWalk()
        g=calcG(r)
        # value[3]=value[3]+alpha*(g[0]-value[3])
        for i in range(len(s)-1):
            ss=s[i]-1
            value[ss]=value[ss]+alpha*(g[i]-value[ss])
        rms0+=rms(value,vT)
        rmsL.append(rms0/(h+1))
    return rmsL
def trainTD(alpha,n=5000,rtV=False):
    vT = [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
    value=np.ones((5))/2
    rms0=0
    rmsL=[]
    for h in range(n):
        s,_,r=randowWalk()
        # g=calcG(r)
        # value[3]=value[3]+alpha*(g[0]-value[3])
        for i in range(len(s)-1):
            ss=s[i]-1
            sss1=s[i+1]
            if sss1==0 or sss1==6:
                value[ss] = value[ss] + alpha * (r[i]  - value[ss])
            else:
                value[ss]=value[ss]+alpha*(r[i]+value[sss1-1]-value[ss])
        rms0+=rms(value,vT)
        rmsL.append(rms0/(h+1))
    if rtV:
        return rmsL,value
    return rmsL



vT=[1/6,2/6,3/6,4/6,5/6]
alpha = 0.01
rmsL=trainMC(alpha)
plt.plot(rmsL,label='MC alpha= 0.01')

alpha = 0.02
rmsL=trainMC(alpha)
plt.plot(rmsL,label='MC alpha= 0.02')

alpha = 0.03
rmsL=trainMC(alpha)
plt.plot(rmsL,label='MC alpha= 0.03')

alpha = 0.04
rmsL=trainMC(alpha)
plt.plot(rmsL,label='MC alpha= 0.04')
#
# plt.legend()
# plt.show()




alpha = 0.15
rmsL=trainTD(alpha)
plt.plot(rmsL,label='TD alpha= 0.15')

alpha = 0.1
rmsL=trainTD(alpha)
plt.plot(rmsL,label='TD alpha= 0.1')

alpha = 0.05
rmsL=trainTD(alpha)
plt.plot(rmsL,label='TD alpha= 0.05')

plt.legend()
plt.show()

alpha=0.1
v0=np.ones((5))/2
plt.plot(v0,label='TD nIter=0')
_,v1=trainTD(alpha,n=1,rtV=True)
plt.plot(v1,label='TD nIter=1')
_,v2=trainTD(alpha,n=10,rtV=True)
plt.plot(v2,label='TD nIter=10')
_,v3=trainTD(alpha,n=100,rtV=True)
plt.plot(v3,label='TD nIter=100')
plt.plot(vT,label='TD vTrue')
plt.legend()
plt.show()























