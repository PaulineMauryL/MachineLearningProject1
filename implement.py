# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
from other import batch_iter
from other import plot_train_test

# -----------------------------------------------------------
# -------------------- Ridge regression ---------------------

def compute_loss_ridge(y, tx, w, lambda_):
    """Calculate the loss of ridge regression."""
    err = y - tx.dot(w)
    loss = np.sqrt(np.mean(err**2))
    return loss


def compute_gradient_ridge(y, tx, w, lambda_):
    """Compute the gradient of ridge regression."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err) + (2 * lambda_* w)
    return grad, err


def ridge_regression(y, tx, lambda_):
    """Compute Ridge regression."""
    lambd = 2 * tx.shape[0] * lambda_
    a = tx.T.dot(tx) + lambd * np.identity(tx.shape[1])
    b = tx.T.dot(y) 
    w = np.linalg.solve(a, b)
    loss = compute_loss_ridge(y, tx, w, lambda_)   
    return w, loss

def ridge_hyperparam(lambdas, nb_fold, nb_crossvalid, x_train, y_train):
    
    loss_valid = np.zeros([len(lambdas), nb_fold])
    loss_train = np.zeros([len(lambdas), nb_fold])
    nb_elem = math.floor(x_train.shape[0]/nb_fold)
    
    for i, lambda_ in enumerate(lambdas):
        for k in range(nb_crossvalid):
            x_valid_k = x_train[k*nb_elem:(k+1)*nb_elem][:]  
            y_valid_k = y_train[k*nb_elem:(k+1)*nb_elem]
            
            x_train_k = np.concatenate([x_train[0:k*nb_elem][:], x_train[(k+1)*nb_elem:][:]])
            y_train_k = np.concatenate([y_train[0:k*nb_elem],    y_train[(k+1)*nb_elem:]   ]) 
                                        
            w, loss_tr = ridge_regression(y_train_k, x_train_k, lambda_)
            loss_train[i][k] = loss_tr
            loss_valid[i][k] = compute_loss_ridge(y_valid_k, x_valid_k, w, lambda_)
    ltrain = np.mean(loss_train, axis=1)
    lvalid = np.mean(loss_valid, axis=1)
    return lvalid, ltrain

# -----------------------------------------------------------
# -------------------- Whole search    ---------------------
def modeling(trx_0,y_train_0,num_intervals_lambda=1,nb_fold=1,nb_crossvalid=1,min_range=-1,max_range=1):

    lambdas_0       = np.logspace(min_range, max_range, num_intervals_lambda)
    valid_r_0, train_r_0= ridge_hyperparam(lambdas_0, nb_fold, nb_crossvalid, trx_0, y_train_0)
    
    plot_train_test(train_r_0, valid_r_0, lambdas_0,2)
    
    # Minimum values for ls_sgd
    ind_0 = np.unravel_index(np.argmin(valid_r_0, axis=None), valid_r_0.shape)
    lambd_0 = lambdas_0[ind_0]
    min_valid_r_loss_0 = valid_r_0[ind_0]
    print("The best lambda is",lambd_0,"with valid test",min_valid_r_loss_0)
    w_best_0,lost_best_0=ridge_regression(y_train_0,trx_0,lambd_0)
    print("And the loss on the whole train data is",lost_best_0)
    
    
    return w_best_0, lambd_0


