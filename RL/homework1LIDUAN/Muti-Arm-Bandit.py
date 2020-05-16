#Author:ike yang
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
def getReward(arms):
    if arms==0:
        return np.random.choice([2, 1, 0], p=[0.2, 0.5, 0.3])
    if arms==1:
        return np.random.chisquare(2)
    if arms==2:
        return 0.5


def maxChose(score):
    a=np.max(score)
    ind=np.where(score==a)[0]
    if len(ind)>1:
        return np.random.choice(ind)
    return ind[0]

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 12,
        }



def makeDecision(Q,type,param=None):
    '''
    :param Q: former reward
    :param type: three strategies
    :param param: corrsponding parameters
    :return:
    '''
    if type=='greedy':
        score=Q
        chosenArms=maxChose(score)
        return chosenArms
    if type=='e-greedy':
        e=param['e']
        p=np.random.random()
        if p<e:
            chosenArms=np.random.choice([2, 1, 0])
        else:
            score=Q
            chosenArms=maxChose(score)
        return chosenArms
    if type=='UCB':
        t=param['t']
        N=param['N']
        c=param['c']
        Nreal=N+0.0001
        score=Q+c*np.sqrt(np.log(t)/Nreal)
        chosenArms=maxChose(score)
        return chosenArms

def updataParam(decision,Q,N,R):
    N[decision] += 1
    Q[decision]+=(R-Q[decision])/N[decision]
    return Q,N


def initParam(arms=3):
    Q=np.zeros([3,])
    N=np.zeros([3,])
    return Q,N

def simulation(maxLoop,method,param=None):

# maxLoop=1000
    if method=='greedy':
        RL1=[]
        RL1ave=[]
        NL1ave=[]

        Q, N = initParam()
        for t in range(maxLoop):
            chosenArms=makeDecision(Q,method)
            R=getReward(chosenArms)
            RL1.append(R)
            RL1ave.append(np.mean(RL1))
            Q, N =updataParam(chosenArms,Q,N,R)
            NL1ave.append(N[1] / np.sum(N))
        # plt.plot(RL1ave,label=method)
        return RL1ave,NL1ave



    if method=='e-greedy':

        if param is None:
            e = 0.1
            param={'e':e}
        else:
            e = param
            param = {}
            param['e'] = e
        RL2=[]
        RL2ave=[]
        NL2ave = []
        Q, N = initParam()
        for t in range(maxLoop):
            chosenArms=makeDecision(Q,method,param=param)
            R=getReward(chosenArms)
            RL2.append(R)
            RL2ave.append(np.mean(RL2))
            Q, N =updataParam(chosenArms,Q,N,R)
            NL2ave.append(N[1] / np.sum(N))
        return RL2ave,NL2ave



    if method=='UCB':
        RL3=[]
        RL3ave=[]
        NL3ave = []
        Q, N = initParam()
        if param is None:
            c = 0.1
            param={}
            param['c']=c
        else:
            c=param
            param = {}
            param['c'] = c
        for t in range(maxLoop):
            param['N']=N
            param['t']=t+1
            chosenArms=makeDecision(Q,method,param=param)
            R=getReward(chosenArms)
            RL3.append(R)
            RL3ave.append(np.mean(RL3))
            Q, N =updataParam(chosenArms,Q,N,R)
            NL3ave.append(N[1] / np.sum(N))
        return RL3ave,NL3ave

# maxloop=5000
# e=0.1
# c=0.3
# r1=simulation(maxloop,'greedy')
# r2=simulation(maxloop,'e-greedy',e)
# r3=simulation(maxloop,'UCB',c)
# plt.plot(r1,label='greedy')
# plt.plot(r2,label='e-greedy e=%f'%(e))
# plt.plot(r3,label='UCB c=%f'%(c))
# plt.legend()
# plt.show()

# maxloop=20
# e=0.0
#
# r2,n2=simulation(maxloop,'e-greedy',e)
# plt.plot(n2,label='e-greedy e=%f'%(e))
#
# e=0.1
# r2,n2=simulation(maxloop,'e-greedy',e)
# plt.plot(n2,label='e-greedy e=%f'%(e))
#
# e=0.01
# r2,n2=simulation(maxloop,'e-greedy',e)
# plt.plot(n2,label='e-greedy e=%f'%(e))
#
# plt.legend()
# plt.show()


# #(a)
maxloop=20
c=0.1
e=0.1
ave1=0
ave2=0
ave3=0
for i in range(500):
    r1,n1=simulation(maxloop,'greedy')
    ave1+=r1[-1]*maxloop
    r2,n2=simulation(maxloop,'e-greedy',e)
    ave2 += r2[-1] * maxloop
    r3,n3=simulation(maxloop,'UCB',c)
    ave3 += r3[-1] * maxloop
ave1=ave1/500
ave2=ave2/500
ave3=ave3/500
print(ave1)
print(ave2)
print(ave3)


#(b)

cL=[]
eL=[]
ave2L=[]
ave3L=[]

maxloop=20
for i in range(90):
    c=0.1+i*0.01
    e=0.1+i*0.01
    cL.append(c)
    eL.append(e)
    ave2=0
    ave3=0
    for i in range(1000):
        r2,n2=simulation(maxloop,'e-greedy',e)
        ave2 += r2[-1] * maxloop
        r3,n3=simulation(maxloop,'UCB',c)
        ave3 += r3[-1] * maxloop
    ave2=ave2/1000
    ave3=ave3/1000
    ave2L.append(ave2)
    ave3L.append(ave3)

f, ax = plt.subplots(1, 1)
# plt.plot(cL,ave3L)
# plt.plot(eL,ave2L)
# plt.show()
ax.set_xlabel('e(c)', font)
ax.set_ylabel('Total Rewards', font)
ax.tick_params(labelsize=7)
ax.plot(eL,ave2L, sns.xkcd_rgb['denim blue'], lw=1.5, label='e-greedy')
ax.plot(cL,ave3L, sns.xkcd_rgb['orange pink'], lw=1.5, label='UCB')


ax.legend(loc='best', prop=font)
plt.show()
f.savefig('%sFina.jpg' % ('b'), dpi=200, bbox_inches='tight')



