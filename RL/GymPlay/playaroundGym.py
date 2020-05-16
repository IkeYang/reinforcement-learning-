import sys
sys.path.append(r"c:\users\user\anaconda3\lib\site-packages")
import gym
import gym
import numpy as np


def initThetaW(observation_P):
    theta=np.random.random((observation_P+1,1))#(s)->(a) (s,1)T*(sa,1)=(1,a)
    w=np.random.random((observation_P+1,1))#(sa,1).T.(sam，1)
    return theta,w

def softm(x):
    y=np.exp(x)
    sum=np.sum(y)
    return y/sum

def sampleAction(observation,theta,a_P): #a=argmaxa(Pi(s))
    a=np.zeros((a_P))
    for i in range(a_P):
        sa=np.hstack((observation.reshape((1,-1)),np.array(i).reshape((1,-1))))
        a[i]=sa.dot(theta)
    pa = softm(a)
    action = np.argmax(pa)
    return action
def getValue(w,observation,a):

    sa=np.hstack((observation.reshape((1,-1)),a.reshape((1,-1))))
    return sa.dot(w)[0,0]

def getDeltP(observation,a,theta,a_P):
    a2 = np.zeros((a_P))
    for i in range(a_P):
        sa = np.hstack((observation.reshape((1, -1)), np.array(i).reshape((1, -1))))
        a2[i] = sa.dot(theta)
    pa = softm(a2)

    sa = np.hstack((observation.reshape((1, -1)), a.reshape((1, -1))))
    E=np.zeros_like(sa)
    for ac in range(a_P):
        sa2 = np.hstack((observation.reshape((1, -1)), np.array(ac).reshape((1, -1))))
        E+=pa[ac]*sa2
    return (sa-E).reshape((-1,1))
env = gym.make('CartPole-v0')
observation_P=env.observation_space.shape[0]
a_P=env.action_space.n
theta,w=initThetaW(observation_P)
gamma=1
alpha=1e-7
beta=1e-7
for i_episode in range(10000):
    observation = env.reset()
    action = sampleAction(observation,theta,a_P)
    for t in range(1000000000):
        # env.render()#画图
        observation2, reward, done, info = env.step(int(action))
        if done:
            reward=-1000
        action2 = sampleAction(observation2, theta,a_P)
        action=np.array(action)
        action2=np.array(action2)
        q1=getValue(w,observation,action)
        q2=getValue(w,observation2,action2)
        d=reward+gamma*q2-q1

        deltP=getDeltP(observation,action,theta,a_P)
        print(reward)
        theta+=alpha*deltP*q1
        sa=np.vstack((observation.reshape((-1, 1)), action.reshape((-1, 1))))
        w+=beta*d*sa
        action=action2
        observation=observation2

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()