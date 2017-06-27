import numpy as np
from Sigmoid import *
from Dropout import *
from Softmax import *


def DeepDropout(W1, W2, W3, W4, X, D):
    alpha = 0.01
    
    N = 5
    for k in range(N):
        x = np.reshape(X[:, :, k], (25, 1))
       
        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        y1 = y1 * Dropout(y1, 0.2)
        
        v2 = np.matmul(W2, y1)
        y2 = Sigmoid(v2)
        y2 = y2 * Dropout(y2, 0.2)
        
        v3 = np.matmul(W3, y2)
        y3 = Sigmoid(v3)
        y3 = y3 * Dropout(y3, 0.2)     
        
        v = np.matmul(W4, y3)
        y = Softmax(v)
        
        d     = D[k,:].T 
        e     = d-y
        delta = e
        
        e3     = np.matmul(W4.T, delta)
        delta3 = y3*(1-y3) * e3
        
        e2     = np.matmul(W3.T, delta3)
        delta2 = y2*(1-y2) * e2
        
        e1     = np.matmul(W2.T, delta2)
        delta1 = y1*(1-y1) * e1
        
        dW4 = alpha * delta * y3.T
        W4  = W4 + dW4
        
        dW3 = alpha * delta3 * y2.T
        W3  = W3 + dW3
        
        dW2 = alpha * delta2 * y1.T
        W2  = W2 + dW2
        
        dW1 = alpha * delta1 * x.T
        W1  = W1 + dW1
    
    return W1, W2, W3, W4
def TestDeepDropout():
    X = np.zeros((5, 5, 5))
    
    X[:, :, 0] = [[0,1,1,0,0],
                  [0,0,1,0,0],
                  [0,0,1,0,0],
                  [0,0,1,0,0],
                  [0,1,1,1,0]]
    
    X[:, :, 1] = [[1,1,1,1,0],
                  [0,0,0,0,1],
                  [0,1,1,1,0],
                  [1,0,0,0,0],
                  [1,1,1,1,1]]
    
    X[:, :, 2] = [[1,1,1,1,0],
                  [0,0,0,0,1],
                  [0,1,1,1,0],
                  [0,0,0,0,1],
                  [1,1,1,1,0]]
    
    X[:, :, 3] = [[0,0,0,1,0],
                  [0,0,1,1,0],
                  [0,1,0,1,0],
                  [1,1,1,1,1],
                  [0,0,0,1,0]]
    
    X[:, :, 4] = [[1,1,1,1,1],
                  [1,0,0,0,0],
                  [1,1,1,1,0],
                  [0,0,0,0,1],
                  [1,1,1,1,0]]
    
    D = np.array([[[1,0,0,0,0]],
                  [[0,1,0,0,0]],
                  [[0,0,1,0,0]],
                  [[0,0,0,1,0]],
                  [[0,0,0,0,1]]])
    
    W1 = 2*np.random.random((20, 25)) - 1
    W2 = 2*np.random.random((20, 20)) - 1
    W3 = 2*np.random.random((20, 20)) - 1
    W4 = 2*np.random.random(( 5, 20)) - 1
    
    
    for _epoch in range(20000):
        W1, W2, W3, W4 = DeepDropout(W1, W2, W3, W4, X, D)
        
    N = 5
    for k in range(N):
        x  = np.reshape(X[:, :, k], (25, 1))
       
        v1 = np.matmul(W1, x)
        y1 = Sigmoid(v1)
        
        v2 = np.matmul(W2, y1)
        y2 = Sigmoid(v2)
        
        v3 = np.matmul(W3, y2)
        y3 = Sigmoid(v3)
        
        v = np.matmul(W4, y3)
        y = Softmax(v)
            
        print("Y = ", k+1, ": ")
        print(y)

if __name__ == '__main__':
    TestDeepDropout()