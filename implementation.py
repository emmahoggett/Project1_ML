import numpy as np

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
    w = np.linalg.solve(A, b)
    err = y - tx.dot(w)
    loss = (1/N) * err.dot(err)
    return w, loss

def ridge_regression(y, tx, lambda_): # modifier 
    """Implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y,tx,w)
    return w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for i in range(max_iters):
        sigma = np.divide(np.exp(np.dot(tx,w)), 1 + np.exp(np.dot(tx,w)))
        grad = np.dot(tx.T,sigma-y)
        loss = - np.dot(y.T,np.log(sigma)) + np.dot((1-y).T, np.log(1-sigma))
        w = w - gamma*grad  
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for i in range(max_iters):
        sigma = np.divide(np.exp(np.dot(tx,w)), 1+np.exp(np.dot(tx,w)))
        grad = np.dot(tx.T,sigma-y)+lambda_*w
        loss = - np.dot(y.T,np.log(sigma))+np.dot((1-y).T, np.log(1-sigma))+lambda_/2*w.T.dot(w)
        w = w - gamma*grad
    return w, loss

#############################################

### for MSE - modifier compute loss

def compute_loss(y, tx, w):
    """Compute the loss with MSE."""
    e = y - tx.dot(w) #error    
    return 1/2*np.mean(e**2)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    e = y - tx.dot(w) #error
    grad = - (1/N) * tx.T.dot(e)
    return grad


