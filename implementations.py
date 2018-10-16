# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import numpy as np
from losses_gradients import compute_loss_ls, compute_gradient_least_squares, compute_loss_ridge, compute_gradient_ridge

def least_squares_GD(y, tx, w_initial, max_iters, gamma):
    """Gradient descent algorithm with least squares."""
    """Required by project description"""
    w = w_initial
    for n_iter in range(max_iters):
        # compute gradient and error
        grad = compute_gradient_least_squares(y, tx, w)
        #print(grad)
        # gradient w by descent update
        w = w - (gamma * grad)
        # print(w)
        # calculate loss    
    loss = compute_loss_ls(y, tx, w)           
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic Gradient Descent algorithm with least squares."""
    """Required by project description"""
    w = initial_w
    batch_size = 1
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad = compute_gradient_least_squares(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
    loss = compute_loss_ls(y, tx, w)                                 #TO CHECK p.3 least squares
    return w, loss

def least_squares(y, tx):
    """Compute the optimal w and the loss with least square technique"""
    """Required by project description"""
    a = tx.T.dot(tx) 
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)                #p.7 du cours least squares
    loss = compute_loss_ls(y, tx, w)                 #p.3 du cours least squares
    return w, loss

# Je suis un peu confuse. Est ce que ce qu'il veut c'est ça ou bien ce que j'ai fait aux fonctions ridge_GD et ridge_SGD ?? 
# ridge_GD et ridge_SGD sont juste en-dessous
def ridge_regression(y, tx, lambda_):
    """Ridge regression."""
    """Required by project description"""
    lambd = 2 * tx.shape[0] * lambda_
    a = tx.T.dot(tx) + lambd * np.identity(tx.shape[1])
    b = tx.T.dot(y) 
    w = np.linalg.solve(a, b) 
    loss = compute_loss_ridge(y, tx, w, lambda_)   
    return w, loss

def ridge_GD(y, tx, initial_w, max_iters, gamma, lambda_):
    """Gradient descent algorithm with Ridge."""
    """Not required specifically""" 
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and error
        grad, _ = compute_gradient_ridge(y, tx, w, lambda_)
        #print("grad is {} \n of shape {}".format(grad, grad.shape))
        #print("\n err of least squares is {}".format(err))
        # gradient w by descent update
        w = w - gamma * grad
        # calculate loss
    loss = compute_loss_ridge(y, tx, initial_w, lambda_)            
    return w, loss

def ridge_SGD(y, tx, initial_w, max_iters, gamma, lambda_):
    """Stochastic Gradient Descent algorithm with least squares."""
    """Not required specifically""" #Implicitely requested I think
    w = initial_w
    batch_size = 1
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_gradient_ridge(y_batch, tx_batch, w, lambda_)    
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
    loss = compute_loss_ridge(y, tx, initial_w, lambda_) 
    return w, loss
'''
def logistic_regression(y,tx,initial_w,max_iters,gamma)

    return w,loss

def reg_logistic_regression(y,tx,lambda_,initial_w,max_iters,gamma)

    return w,loss
'''

#besoin pour SGD, prise de lab02
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