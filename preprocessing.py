import numpy as np

def standardize_train(x):
    ''' standardize training set
    '''
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(x, axis=0)
    
    return std_data, np.mean(x, axis=0), np.std(x, axis=0)

def standardize_valid(x, mean, std):
    ''' standardize test set with same values as training set
    '''
    return (x-mean)/std

def add_bias(x):
    '''add column vector of one for bias'''
    vect_one = np.ones([x.shape[0],1])
    return np.concatenate((vect_one, x), axis = 1)

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1,degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly