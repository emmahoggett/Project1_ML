def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        grad = compute_gradient(y, tx, w)
        w = w - gamma*grad;
        #print("Gradient Descent({bi}/{ti}): loss={l}, w={w}".format(
              #bi=n_iter, ti=max_iters - 1, l=loss, w=w))
    return w, loss