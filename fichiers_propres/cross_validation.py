from implementation import *
from feature_expansion import *

def degree_cross_validation(y, tx, k_fold, max_degree, seed):
    """Finds the optimal degree for the polynomial feature expansion of every feature. The optimal degree is found by comparing the test RMSE (which is computed by cross-validation) for every degree up to max_degree."""
    
    expansion_degrees = np.ones(tx.shape[1])
    tx_expanded = np.c_[tx, np.ones(tx.shape[0])] # We start by adding an offset feature
    
    for feature_index in range(expansion_degrees.shape[0]-1, -1, -1):
        
        list_mse_tr = []
        list_mse_te = []
        
        for degree in range(1, max_degree+1):
            
            # Expand the feature to the desired degree
            tx_expanded_temp = build_poly_index(tx_expanded, degree, feature_index)
            
            # Compute the RMSEs for training and test by cross-validation
            mse_tr = 0
            mse_te = 0
            k_indices = build_k_indices(y, k_fold, seed)  # split data in k fold
            for k in range(k_fold):
                
                # Test on k'th subgroup
                te_indices = k_indices[k]
                tx_te = tx_expanded_temp[te_indices]
                y_te = y[te_indices]
                
                # Train on the other subgroups
                tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)]
                tr_indices = tr_indices.reshape(-1)
                tx_tr = tx_expanded_temp[tr_indices]
                y_tr = y[tr_indices]
                
                tx_tr = build_poly(tx_tr, degree)
                tx_te = build_poly(tx_te, degree)
                
                w, loss_tr = least_squares(y_tr, tx_tr)
                loss_te = compute_loss(y_te, tx_te, w)
                
                mse_tr += loss_tr
                mse_te += loss_te
            
            mse_tr = mse_tr/k_fold
            mse_te = mse_tr/k_fold
            
            list_mse_te.append(mse_te)
            list_mse_tr.append(mse_tr)
            
        # Select the optimal degree for the polynomial expansion of this feature
        optimal_degree = np.argmin(list_mse_te)+1
        expansion_degrees[feature_index] = optimal_degree
        
        # We keep the version of tx expanded to the optimal degree
        tx_expanded  = build_poly_index(tx_expanded, optimal_degree, feature_index)
    
    expansion_degrees = expansion_degrees.astype(int) # Convert to integer
    
    return expansion_degrees, tx_expanded



def build_k_indices(y, k_fold, seed):
    # build k indices for k-fold 
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)