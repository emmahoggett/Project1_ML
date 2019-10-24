import numpy as np

def build_poly(x, degree):
    """ Creates the polynomial basis functions for input data x with degrees going from 1 to 'degree'."""
    poly = x
    
    for deg in range(2, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
        
    return poly



def build_poly_index(x, degree, feature_index):
    
    tx = np.c_[x[:, :feature_index], build_poly(x[:, feature_index], degree), x[:, feature_index+1:]]
    
    return tx



def build_multi_poly(x, degrees):
    
    for i in reversed(range(len(degrees))):
        x = np.c_[x[:, :i], build_poly(x[:, i], degrees[i]), x[:, i+1:]]
        
    tx = np.c_[np.ones((x.shape[0], 1)), x]
    
    return tx


################


def build_poly_data(tX_tr,tX_te,tX_fin,degree):
    tX_tr=build_multi_poly(tX_tr, degree)
    tX_te=build_multi_poly(tX_te, degree)
    tX_fin=build_multi_poly(tX_fin, degree)
    return tX_tr,tX_te,tX_fin