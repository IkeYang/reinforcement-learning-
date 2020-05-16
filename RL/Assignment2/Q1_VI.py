#Author:ike yang
import numpy as np





def initConfig(e):
# e=0.5,0.001
    v0=np.array((0,0,0)).reshape((3,1))
    pss=np.array(([0.75,0.25,0],[0.25,0.75-e,e],[0,e,1-e]))
    return v0,pss


def basicIter(v,pss,gamma):
    vout=np.array((1,2,3)).reshape((3,1))+gamma*pss.dot(v)
    return vout



for e in [0.5,0.01]:
    for gamma in [0.9,0.999]:
        print('e= %f, gamma= %f'%(e,gamma))
        v0, pss=initConfig(e)
        vstart=np.linalg.inv(np.eye(3)-gamma*pss).dot(np.array((1,2,3)))
        print(vstart)
        i=0
        while(1):
            v1=basicIter(v0,pss,gamma)
            i+=1
            error=np.max(v1-v0)
            # print(error)
            if error<0.001:
                print('converged i=%d'%(i),'v= ',v1.flatten())
                break
            v0=v1
















































