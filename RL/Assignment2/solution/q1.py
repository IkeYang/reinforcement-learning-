import numpy as np

gamma_list   = [0.9,0.999]
epsilon_list = [0.5,0.01]

c = np.array([[1],[2],[3]])
error = 1.0e-10

def Bellman_operator():
   for gamma in gamma_list:
      for epsilon in epsilon_list:
         B =np.array([[3/4, 1/4, 0],[1/4, (3/4)-epsilon,epsilon],[0, epsilon, 1-epsilon]])
         V=np.array([[0],[0],[0]])
         while True:
             _value = c + gamma * (np.dot(B,V))
             if sum(abs(_value-V))<=error:                
                 break
             else:
                V = _value
         print('gamma =:', gamma) 
         print ('epsilon=:', epsilon)
         print(V)
                
def main():
    Bellman_operator()

if __name__ == "__main__":
    main()                  