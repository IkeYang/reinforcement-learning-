import sys
sys.path.append(r"c:\users\user\anaconda3\lib\site-packages")
import gym
import gym
import numpy as np
import matplotlib.pyplot as plt

def initThetaW(observation_P,a_P):
    theta=np.random.random((observation_P,a_P))#(s)->(a) (s,1)T*(sa,1)=(1,a)
    w=np.random.random((observation_P+1,1))#(sa,1).T.(sam，1)
    return theta,w

def softm(x):
    y=np.exp(x)
    sum=np.sum(y)
    return y/sum

def sampleAction(observation,theta): #a=argmaxa(Pi(s))
    pa = softm(observation.reshape((1,-1)).dot(theta))
    action = np.argmax(pa)
    return action
def getValue(w,observation,a):

    sa=np.hstack((observation.reshape((1,-1)),a.reshape((1,-1))))
    return sa.dot(w)[0,0]

def getDeltP(observation,a,theta,a_P):
    pa = softm(observation.reshape((1, -1)).dot(theta)).flatten()
    dT = np.zeros_like(theta)
    for i in range(a_P):
        if i==a:
            dT[:,i]=observation.flatten()-pa[i]*observation.flatten()
        else:
            dT[:, i] = - pa[i] * observation.flatten()
    return dT



env = gym.make('CartPole-v0')
observation_P=env.observation_space.shape[0]
a_P=env.action_space.n
theta,w=initThetaW(observation_P,a_P)
gamma=1
alpha=1e-3
beta=1e-3
tL=[]
tM=0
updata=True
for i_episode in range(5000):
    observation = env.reset()
    action = sampleAction(observation,theta)
    for t in range(1000000000):
        if updata==False:
            env.render()#画图
        observation2, reward, done, info = env.step(int(action))
        # if done:
        #     reward=-1000
        action2 = sampleAction(observation2, theta)
        action=np.array(action)
        action2=np.array(action2)
        q1=getValue(w,observation,action)
        q2=getValue(w,observation2,action2)
        d=reward+gamma*q2-q1

        deltP=getDeltP(observation,action,theta,a_P)
        # print(reward)
        if updata:
            theta+=alpha*deltP*q1
            sa=np.vstack((observation.reshape((-1, 1)), action.reshape((-1, 1))))
            w+=beta*d*sa
        action=action2
        observation=observation2

        if done:
            # if t==199:
            #     updata=False
            print("Episode finished after {} timesteps".format(t+1))
            print(i_episode)
            tL.append(t)
            break
env.close()
plt.plot(tL)
plt.show()