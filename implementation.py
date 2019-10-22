import numpy as np


##### MACHINE LEARNING METHODS #####

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w) #compute loss with MSE formula
        grad = compute_gradient(y, tx, w)
        w = w - gamma*grad;
    return w, loss

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        g = 0
        loss = 0
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            loss = loss + compute_loss(minibatch_y, minibatch_tx, w)
            g = g + compute_gradient(minibatch_y, minibatch_tx, w)
        g = (1/batch_size)*g    
        w = w - gamma*g
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

def logistic_regression(y, tx,init_w0, max_iters, gamma,threshold= 1e-8):
    w =init_w0
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

def reg_logistic_regression(y, tx,init_w0, lambda_, max_iters, gamma,threshold= 1e-8):
    w = init_w0
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

#############################################
##### COSTS #####

def compute_loss(y, tx, w):
    """Compute the loss with MSE."""
    N = y.shape[0]
    err = y - tx.dot(w) #error
    loss = (1/(2*N)) * err.dot(err)
    return loss

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    e = y - tx.dot(w) #error
    grad = - (1/N) * tx.T.dot(e)
    return grad

def sigmoid(tx,w):
    return 1/ (1+np.exp(-np.dot(tx,w)))

