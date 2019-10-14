import numpy as np

##### Exploratory data analysis #####

# Extract the data points with the same number of jets 

def extract_jet_num(jet_num, y, tX, ids):
    is_jet_num = np.where(tX[:,22] == jet_num, True, False)
    new_y = y[is_jet_num] #We only keep the training points with the desidered jet_num value
    tX_jet_num = tX[is_jet_num,:]
    #is_not_999 = np.where(tX_jet_num[0,:] == -999, False, True)
    if (jet_num == 0):
        is_not_999 = np.delete(np.arange(30), [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29])
    elif (jet_num == 1):
        is_not_999 = np.delete(np.arange(30), [4, 5, 6, 8, 12, 22, 25, 26, 27, 28])
    else: 
        is_not_999 = np.delete(np.arange(30), [8, 22, 25, 28])
    new_tX = tX_jet_num[:,is_not_999] #We take out the values at -999
    new_ids = ids[is_jet_num]
    return new_y, new_tX, new_ids, tX_jet_num


#Standadize the original dataset 

def standardize(x):
    x = x - np.mean(x, axis=0)
    x = x / np.std(x, axis=0)
    return x
