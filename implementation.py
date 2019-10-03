def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w=np.linalg.solve(a, b)
    err=y-np.dot(tx,w)
    loss=1/tx.shape[1]*np.dot(err.T,err)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w=np.linalg.solve(a, b)
    err=y-np.dot(tx,w)
    loss=1/tx.shape[1]*np.dot(err.T,err)
    return w, loss

