#########
# This file was used for the submission with ID number 23637.
# Categorical accuracy: 0.808
# F1-Score: 0.708
#########


# Useful starting lines

import numpy as np
import matplotlib.pyplot as plt




## Load the training data into feature matrix, class labels, and event ids:

from proj1_helpers import *

DATA_TRAIN_PATH = "data/train.csv" # download train data and supply path
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = "data/test.csv" #download data to predict and supply path
y_pred, tX_pred, ids_pred = load_csv_data(DATA_TEST_PATH)



## Data pre-processing

from data_analysis import *

# Split the data into train and test, so we can test ourselves the accuracy of our model. We usually set `ratio` to 0.8.

# If we want to make an actual submission file, we will train on all our data and therefore `ratio` will be set to 1.

ratio = 1
tX_tr, tX_te, y_tr, y_te, ids_tr, ids_te = split_data(tX, y, ids, ratio)

# Perform the following operations:
# 1. Split every dataset into 8 different groups depending on the value of the `PRI_jet_num` parameter in column 22 and on whether the `DER_mass_MMC` parameter is defined in column 0.
# 2. Get rid of undetermined columns (-999). *Example: if `PRI_jet_num=0`, `DER_deltaeta_jet_jet` (column 4) will always be undetermined*
# 3. Standardize every column (subtract the mean and divide by the standard deviation).
# 4. Delete the training points containing outlier values for some of the features.

jet_num = 0
# d: defined DER_mass_MMC, u: undefined mass
y0d_tr, tX0d_tr, ids0d_tr, y0d_te, tX0d_te, ids0d_te, y0d_pred, tX0d_pred, ids0d_pred = data_processing(jet_num, 'defined', y_tr, tX_tr, ids_tr, y_te, tX_te, ids_te, y_pred, tX_pred, ids_pred)
y0u_tr, tX0u_tr, ids0u_tr, y0u_te, tX0u_te, ids0u_te, y0u_pred, tX0u_pred, ids0u_pred = data_processing(jet_num, 'undefined', y_tr, tX_tr, ids_tr, y_te, tX_te, ids_te, y_pred, tX_pred, ids_pred)

jet_num = 1
y1d_tr, tX1d_tr, ids1d_tr, y1d_te, tX1d_te, ids1d_te, y1d_pred, tX1d_pred, ids1d_pred = data_processing(jet_num, 'defined', y_tr, tX_tr, ids_tr, y_te, tX_te, ids_te, y_pred, tX_pred, ids_pred)
y1u_tr, tX1u_tr, ids1u_tr, y1u_te, tX1u_te, ids1u_te, y1u_pred, tX1u_pred, ids1u_pred = data_processing(jet_num, 'undefined', y_tr, tX_tr, ids_tr, y_te, tX_te, ids_te, y_pred, tX_pred, ids_pred)

jet_num = 2
y2d_tr, tX2d_tr, ids2d_tr, y2d_te, tX2d_te, ids2d_te, y2d_pred, tX2d_pred, ids2d_pred = data_processing(jet_num, 'defined', y_tr, tX_tr, ids_tr, y_te, tX_te, ids_te, y_pred, tX_pred, ids_pred)
y2u_tr, tX2u_tr, ids2u_tr, y2u_te, tX2u_te, ids2u_te, y2u_pred, tX2u_pred, ids2u_pred = data_processing(jet_num, 'undefined', y_tr, tX_tr, ids_tr, y_te, tX_te, ids_te, y_pred, tX_pred, ids_pred)

jet_num = 3
y3d_tr, tX3d_tr, ids3d_tr, y3d_te, tX3d_te, ids3d_te, y3d_pred, tX3d_pred, ids3d_pred = data_processing(jet_num, 'defined', y_tr, tX_tr, ids_tr, y_te, tX_te, ids_te, y_pred, tX_pred, ids_pred)
y3u_tr, tX3u_tr, ids3u_tr, y3u_te, tX3u_te, ids3u_te, y3u_pred, tX3u_pred, ids3u_pred = data_processing(jet_num, 'undefined', y_tr, tX_tr, ids_tr, y_te, tX_te, ids_te, y_pred, tX_pred, ids_pred)



## Polynomial feature expansion

from implementation import *
from optimization import * 
from proj1_helpers import *

# For every feature, determine the optimal degree for the polynomial feature expansion by 4-fold cross-validation on the training dataset.

deg0d = np.ones(tX0d_tr.shape[1],np.int64)
deg1d = np.ones(tX1d_tr.shape[1],np.int64)
deg2d = np.ones(tX2d_tr.shape[1],np.int64)
deg3d = np.ones(tX3d_tr.shape[1],np.int64)

deg0u = np.ones(tX0u_tr.shape[1],np.int64)
deg1u = np.ones(tX1u_tr.shape[1],np.int64)
deg2u = np.ones(tX2u_tr.shape[1],np.int64)
deg3u = np.ones(tX3u_tr.shape[1],np.int64)

for feat_ind in np.arange(len(deg0d)):
    deg0d[feat_ind] =  cross_validation_degree(y0d_tr, tX0d_tr, feat_ind, deg0d)
    
for feat_ind in np.arange(len(deg1d)):
    deg1d[feat_ind] =  cross_validation_degree(y1d_tr, tX1d_tr, feat_ind, deg1d)
    
for feat_ind in np.arange(len(deg2d)):
    deg2d[feat_ind] =  cross_validation_degree(y2d_tr, tX2d_tr, feat_ind, deg2d)
    
for feat_ind in np.arange(len(deg3d)):
    deg3d[feat_ind] =  cross_validation_degree(y3d_tr, tX3d_tr, feat_ind, deg3d)

for feat_ind in np.arange(len(deg0u)):
    deg0u[feat_ind] =  cross_validation_degree(y0u_tr, tX0u_tr, feat_ind, deg0u)
    
for feat_ind in np.arange(len(deg1u)):
    deg1u[feat_ind] =  cross_validation_degree(y1u_tr, tX1u_tr, feat_ind, deg1u)
    
for feat_ind in np.arange(len(deg2u)):
    deg2u[feat_ind] =  cross_validation_degree(y2u_tr, tX2u_tr, feat_ind, deg2u)
    
for feat_ind in np.arange(len(deg3u)):
    deg3u[feat_ind] =  cross_validation_degree(y3u_tr, tX3u_tr, feat_ind, deg3u)

# Expand all the feature matrices.

tX0d_tr = expand(tX0d_tr, deg0d)
tX0d_te = expand(tX0d_te, deg0d)
tX0d_pred = expand(tX0d_pred, deg0d)

tX1d_tr = expand(tX1d_tr, deg1d)
tX1d_te = expand(tX1d_te, deg1d)
tX1d_pred = expand(tX1d_pred, deg1d)

tX2d_tr = expand(tX2d_tr, deg2d)
tX2d_te = expand(tX2d_te, deg2d)
tX2d_pred = expand(tX2d_pred, deg2d)

tX3d_tr = expand(tX3d_tr, deg3d)
tX3d_te = expand(tX3d_te, deg3d)
tX3d_pred = expand(tX3d_pred, deg3d)

tX0u_tr = expand(tX0u_tr, deg0u)
tX0u_te = expand(tX0u_te, deg0u)
tX0u_pred = expand(tX0u_pred, deg0u)

tX1u_tr = expand(tX1u_tr, deg1u)
tX1u_te = expand(tX1u_te, deg1u)
tX1u_pred = expand(tX1u_pred, deg1u)

tX2u_tr = expand(tX2u_tr, deg2u)
tX2u_te = expand(tX2u_te, deg2u)
tX2u_pred = expand(tX2u_pred, deg2u)

tX3u_tr = expand(tX3u_tr, deg3u)
tX3u_te = expand(tX3u_te, deg3u)
tX3u_pred = expand(tX3u_pred, deg3u)



## Machine learning

lambda_ = 0.15

w0d, loss0d = ridge_regression(y0d_tr, tX0d_tr, lambda_)
w1d, loss1d = ridge_regression(y1d_tr, tX1d_tr, lambda_)
w2d, loss2d = ridge_regression(y2d_tr, tX2d_tr, lambda_)
w3d, loss3d = ridge_regression(y3d_tr, tX3d_tr, lambda_)

w0u, loss0u = ridge_regression(y0u_tr, tX0u_tr, lambda_)
w1u, loss1u = ridge_regression(y1u_tr, tX1u_tr, lambda_)
w2u, loss2u = ridge_regression(y2u_tr, tX2u_tr, lambda_)
w3u, loss3u = ridge_regression(y3u_tr, tX3u_tr, lambda_)



## Generate predictions

# Predict the labels with the 8 different models for every different value of `PRI_jet_num`.

y0d_pred = predict_labels(w0d, tX0d_pred)
y1d_pred = predict_labels(w1d, tX1d_pred)
y2d_pred = predict_labels(w2d, tX2d_pred)
y3d_pred = predict_labels(w3d, tX3d_pred)

y0u_pred = predict_labels(w0u, tX0u_pred)
y1u_pred = predict_labels(w1u, tX1u_pred)
y2u_pred = predict_labels(w2u, tX2u_pred)
y3u_pred = predict_labels(w3u, tX3u_pred)

# Create the submission file.

y_pred = np.concatenate([y0d_pred, y1d_pred, y2d_pred, y3d_pred, y0u_pred, y1u_pred, y2u_pred, y3u_pred])
ids_pred = np.concatenate([ids0d_pred, ids1d_pred, ids2d_pred, ids3d_pred, ids0u_pred, ids1u_pred, ids2u_pred, ids3u_pred])

OUTPUT_PATH = 'data/results_ridge2.csv' #name of output file for submission
create_csv_submission(ids_pred, y_pred, OUTPUT_PATH)