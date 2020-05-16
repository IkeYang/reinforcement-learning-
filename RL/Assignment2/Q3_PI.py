#Author:ike yang
import numpy as np
import matplotlib.pyplot as plt


def updatePsas(pi,ph):
    psas=np.zeros((101,101))
    # psas[list(ph.flatten()]

    pia = pi + np.arange(1, 100, 1).reshape((-1, 1))
    pia[np.where(pia > 100)] = 100
    pia = pia.astype(np.int)

    pib = -pi + np.arange(1, 100, 1).reshape((-1, 1))
    pib[np.where(pib < 0)] = 0
    pib = pib.astype(np.int)

    psas[range(1,100),list(pia.flatten())]=ph
    psas[range(1,100),list(pib.flatten())]=1-ph
    zerosPi=np.where(pi.flatten()==0)[0]
    psas[zerosPi+1,zerosPi+1]=1
    psas[-1,-1]=1
    psas[0,0]=1
    return psas


def initConfig(ph):
    reward = np.zeros((101, 1))
    # reward[-1, 0] = 1

    v0=np.zeros((101,1))
    # v0=np.arange(0,1,0.01).reshape((-1,1))
    v0[-1, 0] = 1
    pi0=np.zeros((99,1))
    psas0=updatePsas(pi0,ph)
    return v0,pi0,psas0,reward

def updatePi(v,psas,pi,reward,j,ph):
    for i in range(99):
        amax=min(i+1,100-i-1)
        if amax==0:
            pi[i, 0] = 0
            continue
        if ph > 0.5:
            add=1e-7
        else:
            add=-1e-7
        # vl=ph*v[i+1:i+amax+1+1,0]+(1-ph)*(v[i+1-amax:i+1+1,0][::-1])
        vl=ph*np.copy(v[i+1+1:i+amax+1+1,0])+(1-ph)*np.copy(v[i+1-amax:i+1,0][::-1])
        #adjustValue
        if j>1:
            if i>20 :
                if i<40:
                    vl[-1] += add
                else:
                    vl[-1] += add
        opta=np.argmax(vl)+1
        pi[i,0]=opta
    return pi

def basicIter(v,pss,reward):
    vout=reward+pss.dot(v)
    return vout

def valueIter(v0,psas0,reward):
    i = 0
    while (i<20):
        v1 = basicIter(v0, psas0, reward)
        i += 1
        v0 = v1
    return v0

def main(ph):
    v0, pi0, psas0, reward = initConfig(ph)
    for i in range(100):
        v1 = valueIter(v0, psas0, reward)
        # v1 = np.linalg.inv(np.eye(101) - psas0).dot(reward)
        pi1 = updatePi(v1, psas0, pi0, reward, i, ph)
        error = np.max(pi1 - pi0)
        v0 = v1
        pi0 = pi1
        psas0 = updatePsas(pi0, ph)
    plt.plot(v0)
    plt.show()
    plt.plot(pi0)
    plt.show()


main(ph=0.4)
main(ph=0.25)
main(ph=0.55)











