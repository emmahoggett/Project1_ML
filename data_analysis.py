import numpy as np


def data_processing(jet_num, mass, y_tr, tX_tr, ids_tr, y_te, tX_te, ids_te, y_pred, tX_pred, ids_pred):
    " Performs all of the data processing steps both on the dataset used for training and on the dataset on which predictions are made, so they have the same structure. "
    
    # Extract the data points with the desired PRI_jet_num and DER_mass_MMC values and get rid of any undefined value (-999).
    y_tr, tX_tr, ids_tr = extract(jet_num, mass, y_tr, tX_tr, ids_tr)
    y_te, tX_te, ids_te = extract(jet_num, mass, y_te, tX_te, ids_te)
    y_pred, tX_pred, ids_pred = extract(jet_num, mass, y_pred, tX_pred, ids_pred)
    
    # Standardize the training data and do the same operation on the two other datasets (subtract by the training mean and divide by the training standard deviation).
    tX_tr, m, std = standardize_tr(tX_tr)
    tX_te = standardize(tX_te, m, std)
    tX_pred = standardize(tX_pred, m, std)
    
    # Extract the data training points containing outliers
    tX_tr, ids_tr, y_tr = delete_outliers(tX_tr, ids_tr, y_tr)
    
    return y_tr, tX_tr, ids_tr, y_te, tX_te, ids_te, y_pred, tX_pred, ids_pred



#######################



def extract(jet_num, mass, y, tX, ids):
    " Extracts the data points with the desired PRI_jet_num and DER_mass_MMC values and gets rid of any undefined value (-999)."
    
    # Only keep the data points with the desired jet_num and mass values
    if mass == 'defined':
        keep_data = np.where((tX[:,22] == jet_num) & (tX[:,0] != -999), True, False)
    elif mass == 'undefined':
        keep_data = np.where((tX[:,22] == jet_num) & (tX[:,0] == -999), True, False)
    new_y = y[keep_data]
    new_tX = tX[keep_data, :]
    new_ids = ids[keep_data]
    
    # Extract all the undefined values.
    if (jet_num == 0):
        if mass == 'defined':
            defined = np.delete(np.arange(30), [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29])
        elif mass == 'undefined':
            defined = np.delete(np.arange(30), [0, 4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29])
    elif (jet_num == 1):
        if mass == 'defined':
            defined = np.delete(np.arange(30), [4, 5, 6, 8, 12, 22, 25, 26, 27, 28])
        elif mass == 'undefined':
            defined = np.delete(np.arange(30), [0, 4, 5, 6, 8, 12, 22, 25, 26, 27, 28])
    else:
        if mass == 'defined':
            defined = np.delete(np.arange(30), [8, 22, 25, 28])
        elif mass == 'undefined':
            defined = np.delete(np.arange(30), [0, 8, 22, 25, 28])
            
    new_tX = new_tX[:, defined]
    
    return new_y, new_tX, new_ids



###################



def delete_outliers(tX,ids,y):
    " Delete the training points containing outlier values for some of the features. "
    
    z = np.abs(tX)
    y = y[(z < 3).all(axis=1)]
    ids = ids[(z < 3).all(axis=1)]
    tX = tX[(z < 3).all(axis=1)]
        
    return tX, ids, y



###################



def standardize_tr(x):
    " Standardizes the training dataset. "
    
    
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x = (x - x_mean)/x_std
    
    return x, x_mean, x_std



def standardize(x, x_mean, x_std):
    " Standardize any dataset with given mean and standard deviation values. "
    
    x = (x - x_mean)/x_std
    
    return x



###################



def split_data(x, y, ids, ratio, seed=1):
    " Splits the training dataset in two with a given ratio."
    
    np.random.seed(seed)
    N = len(y)
    perm = np.random.permutation(N)
    split = int(np.floor(ratio * N))
    index_train = perm[1:split]
    index_test = perm[split:]
    # create split
    x_train = x[index_train]
    x_test = x[index_test]
    y_train = y[index_train]
    y_test = y[index_test]
    ids_train = ids[index_train]
    ids_test = ids[index_test]
    return x_train, x_test, y_train, y_test, ids_train, ids_test



################################

# Feature expansion

def expand_vector(x, degree):
    """ Polynomial expansion of a vector x with degrees ranging from 1 to 'degree'."""
    poly = x
    for deg in range(2, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def expand(x, degrees):
    """ Polynomial expansion of a feature matrix with different degrees for every feature. """
    for i in reversed(range(len(degrees))):
        x = np.c_[x[:,:i], expand_vector(x[:,i],degrees[i]), x[:,i+1:]]
    tx = np.c_[np.ones((x.shape[0], 1)), x]
    return tx
