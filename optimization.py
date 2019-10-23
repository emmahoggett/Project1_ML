import numpy as np
from proj1_helpers import *
from data_analysis_logistic import *
from optimization import*
from implementation import *
<<<<<<< HEAD
from data_analysis import *
from plots import *
=======
from cross_validation_reg_logistic import*
>>>>>>> 467ea57458254020ae3e0460e2febc4d911a2f71

##### Cross validation #####

def cross_validation_lambda(y, tX, k_fold=4):
    seed = 1
    #lambdas = [0, 0.1, 0.15, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5]
    lambdas = np.logspace(-4,6,100)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # cross validation
    for lambda_ in lambdas:
        loss_tr = 0.0
        loss_te = 0.0
        for k in range(k_fold):
            l_tr, l_te = cross_validation(y, tX, k_indices, k, lambda_)
            loss_tr = loss_tr + l_tr
            loss_te = loss_te + l_te
        loss_tr = loss_tr/(k_fold)
        loss_te = loss_te/(k_fold)
        rmse_tr.append(loss_tr)
        rmse_te.append(loss_te)
    optimal_lambda = lambdas[np.argmin(rmse_te)]
    return optimal_lambda

<<<<<<< HEAD
def cross_validation_degree(y, X, feat_ind, expansion_degrees, maxDeg=3, k_fold=4):
=======
def cross_validation_degree(y, tX, feat_ind, expansion_degrees, maxDeg=3, k_fold=4):
>>>>>>> 467ea57458254020ae3e0460e2febc4d911a2f71
    """
    feat_ind : index of the feature over which cross validation is done to find the optimal_degree
    expansion_degrees : indicates to what order should the other features be during the cross validation.
    """
    seed = 1
    degrees = np.arange(1,maxDeg+1)
    k_indices = build_k_indices(y, k_fold, seed)  # split data in k fold
    rmse_tr = []
    rmse_te = []
    # cross validation
    for degree in degrees:
        loss_tr = 0.0
        loss_te = 0.0
        expansion_degrees[feat_ind] = degree # change degree of feature at index 'feat_ind'
<<<<<<< HEAD
        tX = build_multi_poly(X, expansion_degrees)
        for k in range(k_fold):
=======
        tX = build_multi_poly(tX, expansion_degrees)
        for k in range(k_fold):
            print('--feat_ind', feat_ind, ', degree', degree, ', fold', k)
>>>>>>> 467ea57458254020ae3e0460e2febc4d911a2f71
            l_tr, l_te = cross_validation(y, tX, k_indices, k)
            loss_tr = loss_tr + l_tr
            loss_te = loss_te + l_te
        loss_tr = loss_tr/(k_fold)
        loss_te = loss_te/(k_fold)
        rmse_tr.append(loss_tr)
        rmse_te.append(loss_te)
<<<<<<< HEAD
    optimal_degree = np.argmin(rmse_te)+1;
    cross_validation_visualization(degrees, rmse_tr, rmse_te)
    return optimal_degree

def cross_validation(y, tx, k_indices, k, lambda_=0.15):
=======
  
    optimal_degree = np.argmin(losses_te)+1;
    return optimal_degree

def cross_validation(y, tx, k_indices, k,init_w0,lambda_=0.15,gamma=0.5,max_iters=15):
>>>>>>> 467ea57458254020ae3e0460e2febc4d911a2f71
    degree = 3
    # get k'th subgroup in test, others in train
    te_indices = k_indices[k]
    tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indices = tr_indices.reshape(-1)
    tr_indices = tr_indices.astype(np.int64)
    tx_tr = tx[tr_indices]
    y_tr = y[tr_indices]
    tx_te = tx[te_indices]
    y_te = y[te_indices]
<<<<<<< HEAD
    #tx_tr = build_poly(tx_tr, degree)
    #tx_te = build_poly(tx_te, degree)
    weight, loss_tr = ridge_regression(y_tr, tx_tr, lambda_)     # ridge regression
=======
    tx_tr = build_poly(tx_tr, degree)
    tx_te = build_poly(tx_te, degree)
    
    #weight, loss_tr = ridge_regression(y_tr, tx_tr, lambda_)     # ridge regression
    weight, loss_tr = logistic_regression(y0_tr, tX0_tr, init_w0, max_iters, gamma)
>>>>>>> 467ea57458254020ae3e0460e2febc4d911a2f71
    # calculate the loss for train and test data
    rmse_tr = np.sqrt(2 * loss_tr)
    rmse_te = np.sqrt(2 * compute_loss(y_te, tx_te, weight))
    return rmse_tr, rmse_te

def build_k_indices(y, k_fold, seed):
    # build k indices for k-fold 
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def best_degree_logistic_regression(tX_tr, y_tr, degree_max=6, max_iter=150, gamma=0.01):
    deg = np.ones(tX_tr.shape[1],np.int64)
    degrees=range(1,degree_max)
    
    for feat_ind in range(len(deg)):
        rmse_te=[]
        grades=[]
        deg_temp = np.ones(tX_tr.shape[1],np.int64)
        for degree_ in degrees:
            deg_temp[feat_ind]=degree_
            tX_te_tmp=build_multi_poly(tX_te,deg_temp)
            tX_tr_tmp=build_multi_poly(tX_tr,deg_temp)
            
            init_w,_=least_squares(y_tr, tX_tr_tmp)
            
            w, _ = logistic_regression(y_tr, tX_tr_tmp,init_w, max_iter, gamma)

            #sigma=sigmoid(tX_te_tmp,w)
            #loss_te = -np.mean(y_te*np.log(sigma)+(1-y_te)*np.log(1-sigma))
            #rmse_te.append(loss_te)
            y_pred = predict_labels(w,tX_te_tmp)
            res = np.where(y_te[:,] == y_pred[:,], 1, 0)
            grade = np.mean(res)
            grades.append(grade)
        deg[feat_ind]= degrees[np.argmax(grades)]
    return deg

def best_degree_least_squares(tX_te, tX_tr, y_tr,y_te, degree_max=6):
    deg = np.ones(tX_tr.shape[1],np.int64)
    degrees=range(1,degree_max)
    for feat_ind in range(len(deg)):
        rmse_te=[]
        grades=[]
        deg_temp = np.ones(tX_tr.shape[1],np.int64)
        for degree_ in degrees:
            deg_temp[feat_ind]=degree_
            tX_te_tmp=build_multi_poly(tX_te,deg_temp)
            tX_tr_tmp=build_multi_poly(tX_tr,deg_temp)
            w, _ = least_squares(y_tr,tX_tr_tmp)

            #sigma=sigmoid(tX_te_tmp,w)
            #loss_te = -np.mean(y_te*np.log(sigma)+(1-y_te)*np.log(1-sigma))
            #rmse_te.append(loss_te)
            y_pred = predict_labels(w,tX_te_tmp)
            res = np.where(y_te[:,] == y_pred[:,], 1, 0)
            grade = np.mean(res)
            grades.append(grade)
        deg[feat_ind]= degrees[np.argmax(grades)]
    return deg
