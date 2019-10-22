import numpy as np
from proj1_helpers import *
from data_analysis_logistic import *
from optimization import*
from implementation import *

def test_lambda_cst(y, tx ,init_w, max_iters, gamma):
    lambdas=np.logspace(-5,1,20)
    rmse=[]
    for lambda_ in lambdas:
        w, loss = reg_logistic_regression(y, tx ,init_w, lambda_, max_iters, gamma)
        rmse.append(compute_loss_reg_logistic(y, tx, w))
    return lambdas[np.argmin(rmse)]

def cross_validation_lambda_reg_logistic(y, tX, max_iters, gamma, k_fold=4):
    seed = 1
    #lambdas = [0, 0.1, 0.15, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5]
    lambdas = np.logspace(-4,0,5)
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
            l_tr, l_te = cross_validation_reg_logistic(y, tX, k_indices, k, lambda_, max_iters, gamma)
            loss_tr = loss_tr + l_tr
            loss_te = loss_te + l_te
            
        loss_tr = loss_tr/(k_fold)
        loss_te = loss_te/(k_fold)
        
        rmse_tr.append(loss_tr)
        rmse_te.append(loss_te)
        
    optimal_lambda = lambdas[np.argmin(rmse_te)]
    return optimal_lambda

def cross_validation_reg_logistic(y, x, k_indices, k, lambda_, max_iters, gamma):
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
    
    w0=np.zeros(tx_tr.shape[1])
    weight, rmse_tr = reg_logistic_regression(y_tr, tx_tr,w0, lambda_, max_iters, gamma)
    # calculate the loss for train and test data
    rmse_te = compute_loss_reg_logistic(y_te, tx_te, weight)
    return rmse_tr, rmse_te

def cross_validation_deg_reg_logistic(y, tX, max_iters, gamma,lambda_, k_fold=4, max_degree=4):
    seed = 1
    #lambdas = [0, 0.1, 0.15, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5]
    degrees=range(1,max_degree)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    
    
    optimal_deg=np.ones(tX.shape[1],np.int64)
    print(optimal_deg.shape)
    # cross validation
    for ind in range(tX.shape[1]):
        rmse_tr = []
        rmse_te = []
        for degree_ in degrees:
            loss_tr = 0
            loss_te = 0
            optimal_deg[ind]=degree_

            for k in range(k_fold):
                l_tr, l_te = cross_validation_reg_logistic_poly(y, tX, k_indices, k, lambda_, max_iters, gamma,optimal_deg)
                loss_tr = loss_tr + l_tr
                loss_te = loss_te + l_te

            loss_tr = loss_tr/(k_fold)
            loss_te = loss_te/(k_fold)

            rmse_tr.append(loss_tr)
            rmse_te.append(loss_te)

        optimal_deg[ind] = degrees[np.argmin(rmse_te)]
    return optimal_deg

def cross_validation_reg_logistic_poly(y, x, k_indices, k, lambda_, max_iters, gamma,deg):
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
    
    tx_te_tmp=build_multi_poly(tx_te,deg)
    tx_tr_tmp=build_multi_poly(tx_tr,deg)
    
    w0=np.zeros(tx_tr_tmp.shape[1])
    weight, rmse_tr = reg_logistic_regression(y_tr, tx_tr_tmp,w0, lambda_, max_iters, gamma)
    # calculate the loss for train and test data
    rmse_te = compute_loss_reg_logistic(y_te, tx_te_tmp, weight)
    
    return rmse_tr, rmse_te