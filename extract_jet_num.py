def extract_jet_num(jet_num, y, tX, ids):
    
    is_jet_num = tX[:,22] == jet_num
    
    new_y = y[is_jet_num] #We only keep the training points with the desidered jet_num value
    
    tX_jet_num = tX[is_jet_num,:]
    new_tX = tX_jet_num[tX_jet_num[0,:] != -999] #We take out the values at -999
    
    new_ids = ids[is_jet_num]
    
    return new_y, new_tX, new_ids, tX_jet_num