import numpy as np
from Sigmoid import *


def BackpropCE(W1, W2, X, D):
    alpha = 0.9

    N = 4
    for k in range(N):
        xT = X[k, :].T
        d = D[k]
        
        v1 = np.matmul(W1, xT)
        y1 = Sigmoid(v1)
        v  = np.matmul(W2, y1)
        y  = Sigmoid(v)
        
        e     = d - y
        delta = e
        
        e1     = np.matmul(W2.T, delta)
        delta1 = y1*(1-y1) * e1
        
        dW1 = (alpha*delta1).reshape(4, 1) * xT.reshape(1, 3)
        W1  = W1 + dW1
               
        dW2 = alpha * delta * y1
        W2  = W2 + dW2
    
    return W1, W2

def TestBackpropCE():
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    
    D = np.array([[0],
                  [0],
                  [1],
                  [1]])
    
    W1 = 2*np.random.random((4, 3)) - 1
    W2 = 2*np.random.random((1, 4)) - 1
    
    for _epoch in range(10000):
        W1, W2 = BackpropCE(W1, W2, X, D)
        
    N = 4
    for k in range(N):
        x  = X[k, :].T
        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        v  = np.matmul(W2, y1)
        y  = Sigmoid(v)
        print(y)
        
if __name__ == '__main__':
    TestBackpropCE()