def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        g = 0
        loss = 0
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            loss = loss + compute_loss(minibatch_y, minibatch_tx, w)
            g = g + compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        g = (1/batch_size)*g    
        w = w - gamma*g
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              #bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss