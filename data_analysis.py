import numpy as np

##### Exploratory data analysis #####

def data_analysis(jet_num, y_tr, tX_tr, ids_tr, y_te, tX_te, ids_te):
    y_tr, tX_tr, ids_tr, _ = extract_jet_num(jet_num, y_tr, tX_tr, ids_tr)
    y_te, tX_te, ids_te, _ = extract_jet_num(jet_num, y_te, tX_te, ids_te)
    
    tX_tr=extract_values(tX_tr, ids_tr)
    tX_te=extract_values(tX_te, ids_te)
    
    #tX_tr, m, std = standardize(tX_tr)
    #tX_te = standardize_te(tX_te, m, std)
    return y_tr, tX_tr, ids_tr, y_te, tX_te, ids_te

# Extract the data points with the same number of jets

def extract_jet_num(jet_num, y, tX, ids):
    is_jet_num = np.where(tX[:,22] == jet_num, True, False)
    new_y = y[is_jet_num] #We only keep the training points with the desidered jet_num value
    tX_jet_num = tX[is_jet_num,:]
    if (jet_num == 0):
        #is_not_999 = np.delete(np.arange(30), [4, 5, 6, 12, 22, 23, 24, 26, 27, 29])
        is_not_999 = np.delete(np.arange(30), [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29]) 
    elif (jet_num == 1):
        #is_not_999 = np.delete(np.arange(30), [4, 5, 6, 12, 22, 26, 27])
        is_not_999 = np.delete(np.arange(30), [4, 5, 6, 8, 12, 22, 25, 26, 27, 28])
    else: 
        #is_not_999 = np.delete(np.arange(30), [22])
        is_not_999 = np.delete(np.arange(30), [8, 22, 25, 28])
    new_tX = tX_jet_num[:,is_not_999] #We take out the values at -999
    new_ids = ids[is_jet_num]
    return new_y, new_tX, new_ids, tX_jet_num

# Remaining -999 values
def extract_values(tX, ids):
    
    for i in range(tX.shape[1]):
        is_undefined = tX[:,i] == -999
        mean = tX[is_undefined == False, i].mean()
        new_tX = np.copy(tX)
        
        for j in range(new_tX.shape[0]):
            if is_undefined[j]:
                new_tX[j,i] = mean
        
    return new_tX

#Standardize the original dataset 

def standardize(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x = (x - x_mean)/x_std
    return x, x_mean, x_std

#Standadize the test dataset of the same mean and sigma as the train dataset

def standardize_te(x, x_mean, x_std):
    x = (x - x_mean)/x_std
    return x

# Split the original dataset into a train (80%) and a test data (20%)

def split_data(x, y, ids, ratio, seed=1):
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


# Extend the features 

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

