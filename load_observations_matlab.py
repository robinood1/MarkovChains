import numpy as np
import scipy.io
from implementations import *

var = scipy.io.loadmat('observations.mat')
X = scipy.io.loadmat('ground_truth.mat')['X'].flatten()

Y = var['Y']
Y = Y.reshape(Y.size)
W = var['W']
m = int(var['m'])
n = int(var['n'])

print(np.shape(Y))
print(np.shape(W))
print(m)
print(n)

x0 = np.random.randint(low=0, high=2, size=n)*2-1 # A random array of -1 and +1

x_hat,errors,_,_ = metropolis(W,X,x0,0,0.2)
print("ERROR:{}".format(min(errors)))
scipy.io.savemat('JRLCactus', {'x_estimate':x_hat}, appendmat=True, format='5', oned_as='column')