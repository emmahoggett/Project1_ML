import numpy as np


##### MACHINE LEARNING METHODS #####


##### IMPLEMENTATION OF LEAST SQUARES GRADIENT DESCENT #####


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent optimization."""
    
    N = y.shape[0]
    
    # Initialise the output variables.
    w = initial_w
    err = y - tx.dot(w)
    loss = 1/(2*N) * err.dot(err)
    
    for i in range(max_iters):
        err = y - tx.dot(w)
        grad = -tx.T.dot(err) / N # Compute the gradient.
        w = w - gamma*grad # Compute the updated weights.
        loss = 1/(2*N) * err.dot(err) # Compute the new loss value.
    
    return w, loss



##### IMPLEMENTATION OF LEAST SQUARES STOCHASTIC GRADIENT DESCENT #####


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent optimization."""
    
    np.random.seed()
    N = y.shape[0]
    
    # Initialise the output variables.
    w = initial_w
    err = y - tx.dot(w)
    loss = 1/(2*N) * err.dot(err)
    
    for i in range(max_iters):
        
        sample_index = np.random.randint(0, N) # Sample one data point at random
        y_sample = y[sample_index]
        tx_sample = tx[sample_index, :]
        
        err_sample = y_sample - tx_sample.dot(w)
        grad_sample = -tx_sample.T.dot(err_sample) # Compute the gradient of the loss contributed by the sample point.
        w = w - gamma*grad_sample # Compute the updated weights.
        
        err = y - tx.dot(w)
        loss = 1/(2*N) * err.dot(err) # Compute the new loss value.
        
    return w, loss



def least_squares(y, tx):
    """Calculate the least squares solution."""
    N = y.shape[0]
    A = tx.T.dot(tx)
    b = tx.T.dot(y)
    #w = np.linalg.solve(A, b)
    w = np.linalg.lstsq(A, b, rcond=None)[0] #improvement of computation if A singular matrix
    err = y - tx.dot(w)
    loss = (1/N) * err.dot(err)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """Implement ridge regression."""
    aI = lambda_ * np.identity(tx.shape[1])
    A = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    loss = compute_loss(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma,threshold= 1e-8):
    w = initial_w
    losses=[]
    for i in range(max_iters):
        sigma = 1/ (1+np.exp(-np.dot(tx,w)))
        grad = np.dot(tx.T,sigma-y)/y.shape[0]
        loss = -np.mean(y*np.log(sigma)+(1-y)*np.log(1-sigma))
        w = w - gamma*grad  
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        #np.dot(y.T,np.log(sigma))+np.dot((np.ones(y.shape[0])-y).T, np.log(1-sigma))
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma,threshold= 1e-8):
    w = initial_w
    losses=[]
    for i in range(max_iters):
        sigma = 1/ (1+np.exp(-np.dot(tx,w)))
        grad = np.dot(tx.T,sigma-y)/y.shape[0]+lambda_*w
        loss = -np.mean(y*np.log(sigma)+(1-y)*np.log(1-sigma))+lambda_/2*np.dot(w.T,w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        w = w - grad*gamma
    return w, loss




######################

def compute_loss(y, tx, w):
    """Compute the loss with MSE."""
    N = y.shape[0]
    err = y - tx.dot(w) #error
    loss = (1/(2*N)) * err.dot(err)
    return loss