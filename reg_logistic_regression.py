def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    ws=[]
    w=initial_w
    losses=[]
    
    for i in range(max_iters):
        sigma=np.divide(np.exp(np.dot(tx,w)), 1+np.exp(np.dot(tx,w)))
        grad=np.dot(tx.T,sigma-y)+lambda_*w
        loss=-np.dot(y.T,np.log(sigma))+np.dot((1-y).T, np.log(1-sigma))+lambda_/2*w.T.dot(w)
        w=w-gamma*grad
        ws.append(w)
        losses.append(loss)
    
    return w, loss