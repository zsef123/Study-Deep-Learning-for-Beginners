import numpy as np
from scipy import signal
from Softmax import *
from ReLU import *
from Conv import *
from Pool import *


def MnistConv(W1, W5, Wo, X, D):
    alpha = 0.01
    beta  = 0.95
    
    momentum1 = np.zeros_like(W1)
    momentum5 = np.zeros_like(W5)
    momentumo = np.zeros_like(Wo)
    
    N = len(D)
    
    bsize = 100
    blist = np.arange(0, N, bsize)
    
    for batch in range(len(blist)):
        dW1 = np.zeros_like(W1)
        dW5 = np.zeros_like(W5)
        dWo = np.zeros_like(Wo)
        
        begin = blist[batch]
        
        for k in range(begin, begin+bsize):
            # Forward pass = inference     
            x  = X[k, :, :]
            y1 = Conv(x, W1)
            y2 = ReLU(y1)
            y3 = Pool(y2)
            y4 = np.reshape(y3, (-1, 1))
            v5 = np.matmul(W5, y4)
            y5 = ReLU(v5)
            v  = np.matmul(Wo, y5)
            y  = Softmax(v)

            # one-hot encoding
            d = np.zeros((10, 1))
            d[D[k][0]][0] = 1 
            
            # Backpropagation
            e     = d - y
            delta = e
            
            e5     = np.matmul(Wo.T, delta)    # Hidden(ReLU)
            delta5 = (y5 > 0) * e5
            
            e4 = np.matmul(W5.T, delta5)       # Pooling layer
            
            e3 = np.reshape(e4, y3.shape)
            
            e2 = np.zeros_like(y2)             # pooling
            W3 = np.ones_like(y2) / (2*2)            
            for c in range(20):
                e2[:, :, c] = np.kron(e3[:, :, c], np.ones((2, 2))) * W3[:, :, c]
                
            delta2 = (y2 > 0) * e2
            
            delta1_x = np.zeros_like(W1)            
            for c in range(20):
                delta1_x[:, :, c] = signal.convolve2d(x[:, :], np.rot90(delta2[:, :, c], 2), 'valid')
            
            
            dW1 = dW1 + delta1_x
            dW5 = dW5 + np.matmul(delta5, y4.T)
            dWo = dWo + np.matmul(delta, y5.T)
            
        dW1 = dW1 / bsize
        dW5 = dW5 / bsize
        dWo = dWo / bsize
        
        momentum1 = alpha*dW1 + beta*momentum1
        W1        = W1 + momentum1
        
        momentum5 = alpha*dW5 + beta*momentum5
        W5        = W5 + momentum5
        
        momentumo = alpha*dWo + beta*momentumo 
        Wo        = Wo + momentumo
        
    return W1, W5, Wo

def TestMnistConv():
    # Learn
    #
    Images, Labels = LoadMnistData('MNIST\\t10k-images-idx3-ubyte.gz', 'MNIST\\t10k-labels-idx1-ubyte.gz')
    Images = np.divide(Images, 255)

    W1 = 1e-2 * np.random.randn(9, 9, 20)
    W5 = np.random.uniform(-1, 1, (100, 2000)) * np.sqrt(6) / np.sqrt(360 + 2000)
    Wo = np.random.uniform(-1, 1, ( 10,  100)) * np.sqrt(6) / np.sqrt( 10 +  100)

    X = Images[0:8000, :, :]
    D = Labels[0:8000]
        
    for _epoch in range(3):
        print(_epoch)
        W1, W5, Wo = MnistConv(W1, W5, Wo, X, D)

        
    # Test
    #
    X = Images[8000:10000, :, :]
    D = Labels[8000:10000]

    acc = 0
    N   = len(D)
    for k  in range(N):
        x  = X[k, :, :]

        y1 = Conv(x, W1)
        y2 = ReLU(y1)
        y3 = Pool(y2)
        y4 = np.reshape(y3, (-1, 1))
        v5 = np.matmul(W5, y4)
        y5 = ReLU(v5)
        v  = np.matmul(Wo, y5)
        y  = Softmax(v)
        
        i = np.argmax(y)
        if i == D[k][0]:
            acc = acc + 1
            
    acc = acc / N
    print("Accuracy is : ", acc)

if __name__=="__main__":
    TestMnistConv()