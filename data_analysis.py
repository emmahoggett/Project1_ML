import numpy as np

##### Exploratory data analysis #####

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

# Extract the data points with the same number of jets 

def extract_jet_num(jet_num, y, tX, ids):
    is_jet_num = np.where(tX[:,22] == jet_num, True, False)
    new_y = y[is_jet_num] #We only keep the training points with the desidered jet_num value
    tX_jet_num = tX[is_jet_num,:]
    if (jet_num == 0):
        is_not_999 = np.delete(np.arange(30), [4, 5, 6, 12, 23, 24, 26, 27, 29])
        #is_not_999 = np.delete(np.arange(30), [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29]) 
    elif (jet_num == 1):
        is_not_999 = np.delete(np.arange(30), [4, 5, 6, 12, 26, 27])
        #is_not_999 = np.delete(np.arange(30), [4, 5, 6, 8, 12, 22, 25, 26, 27, 28])
    else: 
        is_not_999 = np.arange(30)
        #is_not_999 = np.delete(np.arange(30), [8, 22, 25, 28])
    new_tX = tX_jet_num[:,is_not_999] #We take out the values at -999
    new_ids = ids[is_jet_num]
    return new_y, new_tX, new_ids, tX_jet_num


#Standadize the original dataset 

def standardize(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    print(x_std)
    if (x_std.any == 0):
        x_std = 1
    x = (x - x_mean)/x_std
    return x, x_mean, x_std

#Standadize the test dataset of the same mean and sigma as the train dataset

def standardize_te(x, x_mean, x_std):
    if (x_std.all == 0):
        x_std = 1
    x = (x - x_mean)/x_std
    return x
