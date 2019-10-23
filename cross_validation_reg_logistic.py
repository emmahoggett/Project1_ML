import numpy as np
from proj1_helpers import *
from data_analysis_logistic import *
from optimization import*
from implementation import *

def test_lambda_cst(y, tx,y_te,tx_te ,init_w, max_iters, gamma):
    lambdas=np.logspace(-4,0,30)
    rmse=[]
    for lambda_ in lambdas:
        w,_ = reg_logistic_regression(y, tx ,init_w, lambda_, max_iters, gamma)
        
        y_pred = predict_labels(w, tx_te)
        res = np.where(y_te[:,] == y_pred[:,], 1, 0)
        grade = np.mean(res)
      
        rmse.append(grade)
    return lambdas[np.argmax(rmse)]

def cross_validation_lambda_reg_logistic(y, tX, max_iters, gamma, k_fold=4):
    seed = 1
    #lambdas = [0, 0.1, 0.15, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5]
    lambdas=np.logspace(-4,0,30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_te = []
    # cross validation
    for ind, lambda_ in enumerate(lambdas):
        grade_te = 0
        
        for k in range(k_fold):
            gr_te= cross_validation_reg_logistic(y, tX, k_indices, k, lambda_, max_iters, gamma)
            grade_te = grade_te + gr_te
        
        grade_te = grade_te/(k_fold)
        rmse_te.append(grade_te)
        
    optimal_lambda = lambdas[np.argmax(rmse_te)]
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
    weight,_ = reg_logistic_regression(y_tr, tx_tr,w0, lambda_, max_iters, gamma)
    # calculate the loss for train and test data
    
    y_pred = predict_labels(weight, tx_te)
    res = np.where(y_te[:,] == y_pred[:,], 1, 0)
    grade = np.mean(res)
    
    return grade

def cross_validation_deg_reg_logistic(y, tX, max_iters, gamma,lambda_, k_fold=4, max_degree=5):
    seed = 1
    #lambdas = [0, 0.1, 0.15, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5]
    degrees=range(1,max_degree)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    
    
    optimal_deg=np.ones(tX.shape[1],np.int64)
    # cross validation
    for ind in range(tX.shape[1]):
        rmse_te = []
        for degree_ in degrees:
            grade_te = 0
            optimal_deg[ind]=degree_

            for k in range(k_fold):
                gr_te = cross_validation_reg_logistic_poly(y, tX, k_indices, k, lambda_, max_iters, gamma,optimal_deg)
                grade_te = grade_te + gr_te

            grade_te = grade_te/(k_fold)
            rmse_te.append(grade_te)

        optimal_deg[ind] = degrees[np.argmax(rmse_te)]
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
    weight, _= reg_logistic_regression(y_tr, tx_tr_tmp,w0, lambda_, max_iters, gamma)
    # calculate the loss for train and test data
    y_pred = predict_labels(weight, tx_te_tmp)
    res = np.where(y_te[:,] == y_pred[:,], 1, 0)
    grade = np.mean(res)
    
    return grade