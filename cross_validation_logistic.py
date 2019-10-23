import numpy as np
from proj1_helpers import *
from data_analysis_logistic import *
from implementation import *


def cross_validation_deg_logistic(y, tX, max_iters, gamma, k_fold=4, max_degree=5):
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
                gr_te = cross_validation_logistic_poly(y, tX, k_indices, k, max_iters, gamma,optimal_deg)
                grade_te = grade_te + gr_te

            grade_te = grade_te/(k_fold)
            rmse_te.append(grade_te)

        optimal_deg[ind] = degrees[np.argmax(rmse_te)]
    return optimal_deg

def cross_validation_logistic_poly(y, x, k_indices, k, max_iters, gamma,deg):
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
    weight, _= logistic_regression(y_tr, tx_tr_tmp,w0, max_iters, gamma)
    # calculate the loss for train and test data
    y_pred = predict_labels(weight, tx_te_tmp)
    res = np.where(y_te[:,] == y_pred[:,], 1, 0)
    grade = np.mean(res)
    
    return grade

def build_k_indices(y, k_fold, seed):
    # build k indices for k-fold 
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
