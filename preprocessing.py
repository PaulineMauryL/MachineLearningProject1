import numpy as np

def standardize_train(x):
    ''' standardize training set
    '''
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(x, axis=0)
    
    return std_data, np.mean(x, axis=0), np.std(x, axis=0)

def standardize_test(x, mean, std):
    ''' standardize test set with same values as training set
    '''
    return (x-mean)/std

def add_bias(x):
    '''add column vector of one for bias'''
    vect_one = np.ones([x.shape[0],1])
    return np.concatenate((vect_one, x), axis = 1)