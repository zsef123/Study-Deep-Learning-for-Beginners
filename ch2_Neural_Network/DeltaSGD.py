import numpy as np
from Sigmoid import *


def DeltaSGD(W, X, D):
    alpha = 0.9
    
    N = 4
    for k in range(N):
        x = X[k, :].T
        d = D[k]
    
        v = np.matmul(W, x)
        y = Sigmoid(v)
        
        e     = d - y
        delta = y*(1-y) * e
        
        dW = alpha*delta*x
        
        W[0][0] = W[0][0] + dW[0]
        W[0][1] = W[0][1] + dW[1]
        W[0][2] = W[0][2] + dW[2]
        
    return W
    
def TestDeltaSGD():
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    
    D = np.array([[0],
                  [0],
                  [1],
                  [1]])
        
    W = 2*np.random.random((1, 3)) - 1
        
    for _epoch in range(10000):
        W = DeltaSGD(W, X, D)
                
    N = 4
    for k in range(N):
        x = X[k,:].T
        v = np.matmul(W, x)
        y = Sigmoid(v)
        print(y)

if __name__ == '__main__':
    TestDeltaSGD()