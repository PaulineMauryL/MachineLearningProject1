import numpy as np
import matplotlib.pyplot as plt

import math
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission

def plot_train_test(train_errors, test_errors, lambdas, degree):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    
    degree is just used for the title of the plot.
    """
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(lambdas, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.title("Ridge regression for polynomial degree " + str(degree))
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("ridge_regression")

def cross_validation(x_train,y_train, nb_cross, nb_division):

    nb_elem = math.floor(x_train.shape[0]/nb_division)
    
    for k in range(nb_cross):
        x_valid_k = x_train[k*nb_elem:(k+1)*nb_elem][:]  
        y_valid_k = y_train[k*nb_elem:(k+1)*nb_elem]

        x_train_k = np.concatenate([x_train[0:k*nb_elem][:], x_train[(k+1)*nb_elem:][:]])
        y_train_k = np.concatenate([y_train[0:k*nb_elem],    y_train[(k+1)*nb_elem:]   ])  
    
    return x_train_k,y_train_k,x_valid_k,y_valid_k

def remove_999(input_train, y_train):
    idx = np.isin(input_train, -999.0)
    idx = np.any(idx,axis=1)
    ind = np.nonzero(idx)[0]
    
    x_train_no_999 = np.delete(input_train,ind,axis=0)
    y_train_no_999 = np.delete(y_train,ind)
    
    return  x_train_no_999, y_train_no_999

def remove_999col(input_train,input_test):
    idx = np.isin(input_train, -999.0)
    idx = np.any(idx,axis=0)
    ind = np.nonzero(idx)[0]
    
    x_train_no_999col = np.delete(input_train,ind,axis=1)
    x_test_no_999col = np.delete(input_test,ind,axis=1)
    
    return  x_train_no_999col, x_test_no_999col

def replace_999(input_train):
    idx = np.isin(input_train, -999.0)
    input_train[idx] = 0
    return input_train

def replace_1(input_train):
    idx = np.isin(input_train, -1.0)
    input_train[idx] = 0
    return input_train

def accuracy(y_test, x_test, w):
    y_pred = predict_labels(w, x_test) # Pas de "-", on le voit grace à l'accuracy calculée pour les train data ci-dessous
    #print(y_pred[:10],y_test[:10])
    accuracy = sum(y_pred == y_test)/len(y_test)
    return accuracy

def split_categories(input_tr):
    jet_num = 22
    cat_0 = input_tr[input_tr[:,jet_num] == 0]
    cat_1 = input_tr[input_tr[:,jet_num] == 1]
    cat_2 = input_tr[input_tr[:,jet_num] == 2]
    cat_3 = input_tr[input_tr[:,jet_num] == 3]
    return cat_0, cat_1, cat_2, cat_3

#Taken from lab02 of ML course
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]      
