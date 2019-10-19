def cross_validation_lambda(y, tX, w0, max_iters, gamma, k_fold=4,):
    seed = 1
    #lambdas = [0, 0.1, 0.15, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5]
    lambdas = np.logspace(-4,1,20)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # cross validation
    for ind, lambda_ in enumerate(lambdas):
        loss_tr = 0
        loss_te = 0
        for k in range(k_fold):
            l_tr, l_te = cross_validation(y, tX, k_indices, k, lambda_, w0, max_iters, gamma)
            loss_tr = loss_tr + l_tr
            loss_te = loss_te + l_te
        loss_tr = loss_tr/(k_fold)
        loss_te = loss_te/(k_fold)
        rmse_tr.append(loss_tr)
        rmse_te.append(loss_te)
    optimal_lambda = lambdas[np.argmin(rmse_te)]    
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    return optimal_lambda

def cross_validation(y, x, k_indices, k, lambda_, w0, max_iters, gamma):
    # get k'th subgroup in test, others in train
    te_indices = k_indices[k]
    tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indices = tr_indices.reshape(-1)
    x_tr = x[tr_indices]
    y_tr = y[tr_indices]
    x_te = x[te_indices]
    y_te = y[te_indices]
    tx_tr = x_tr
    tx_te = x_te
    #tx_tr = build_poly(x_tr, degree)
    #tx_te = build_poly(x_te, degree)
    # ridge regression
    weight, _ = reg_logistic_regression(y_tr, tx_tr, lambda_, w0, max_iters, gamma)
    
    # calculate the loss for train and test data
    rmse_tr = np.sqrt(2 * compute_loss(y_tr, tx_tr, weight))
    rmse_te = np.sqrt(2 * compute_loss(y_te, tx_te, weight))
    return rmse_tr, rmse_te
