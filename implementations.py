import numpy as np
import matplotlib.pyplot as plt
import sys as sys

def ReLU(x) :
    return x * (x > 0)

def compute_Y(W, X) :
    n = W.shape[1]
    return ReLU((W @ X)/np.sqrt(n))

# Compute the energy for a given x compared to the truth osbervation
def compute_H(W, x, Y) :
    diff = Y - compute_Y(W, x)
    return diff.T.dot(diff)

# Flip the sign of a random element of x. 
def random_flip(x) :
    i = np.random.randint(low=0, high=len(x))
    new_x = x.copy()
    new_x[i] = -new_x[i]
    return new_x

# Computation of the acceptance probability
def accept(W, x0, x1, beta, Y) :
    return min(1, np.exp(-beta*( compute_H(W, x1, Y) - compute_H(W, x0, Y) )))

# The error between X and x0
def rec_error(x0, X, n):
    return (x0-X).T.dot(x0-X) / (4*n)

def metropolis(W, X, x0, threshold, beta0) :
    #Initialization of the parameters
    m = W.shape[0]
    e = sys.maxsize
    beta = beta0
    Y_true = compute_Y(W, X)
    n = len(X)
    errors = []
    energies = []
    betas = []
    i = 0
    nbre_iter= 10000
    from_last_beta = 0
    beta_augm =0.2
    treshold = nbre_iter/40
    print("Iteration where beta changed:")
    
    #Beginning of the algorithm
    while e > threshold and i < nbre_iter: 
        i +=1
        from_last_beta += 1
        
        # Flip one element of x0 and compute the acceptance probability
        x1 = random_flip(x0)
        a = accept(W, x0, x1, beta, Y_true)
        
        # With probability a, change x0 to x1 for the next iteration
        if a >= np.random.uniform() : 
            x0 = x1
            
        #Increase beta if we are in a stagnation phase AND if w did not increase beta in the last from_last_beta ierations
        if (from_last_beta> treshold and np.std(energies[-200:]) < m/285.7):
            print(i, end=', ')
            beta += beta_augm
            from_last_beta = 0
            
        #Append to the energies and the errors their values for this iteration 
        energies.append(compute_H(W, x0, Y_true))        
        e = rec_error(x0, X, n)
        errors.append(e)      
        betas.append(beta)
        
    return x0, errors, energies, betas
