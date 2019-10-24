import numpy as np

def extract_999(tX):
    for i in range(tX.shape[1]):
        is_defined = np.where(tX[:,i] != -999, True, False)
        tX_defined = tX[is_defined,i]
        mean = tX_defined.mean()
        for j in range(tX.shape[0]):
            if (tX[j,i] == -999):
                tX[j,i] = mean
    return tX

