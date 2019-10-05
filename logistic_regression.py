def logistic_regression(y, tx, initial_w, max_iters, gamma):
    ws=[]
    w=initial_w
    losses=[]
    
    for i in range(max_iters):
        sigma=np.divide(np.exp(np.dot(tx,w)), 1+np.exp(np.dot(tx,w)))
        grad=np.dot(tx.T,sigma-y)
        loss=-np.dot(y.T,np.log(sigma))+np.dot((1-y).T, np.log(1-sigma))
        w=w-gamma*grad
        ws.append(w)
        losses.append(loss)
    
    return w, loss