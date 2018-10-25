# -*- coding: utf-8 -*-
import numpy as np
import math
from other import batch_iter

# -----------------------------------------------------------
# -------------------- Ridge regression ---------------------

# Je suis un peu confuse. Est ce que ce qu'il veut c'est ça ou bien ce que j'ai fait aux fonctions ridge_GD et ridge_SGD ?? 
# ridge_GD et ridge_SGD sont juste en-dessous

def compute_loss_ridge(y, tx, w, lambda_):
    """Calculate the loss of ridge regression."""
    err = y - tx.dot(w)
    loss = (1/2) * np.mean(err**2) + lambda_ * (np.linalg.norm(w,2))**2
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

def ridge_SGD(y, tx, initial_w, max_iters, gamma, lambda_):
    """Stochastic Gradient Descent algorithm with least squares."""
    w = initial_w
    batch_size = 1
    loss0=-1
    
    for n_iter in range(max_iters):
        loss1=compute_loss_ridge(y,tx,w,lambda_)
        if loss0>-1 and loss0<loss1:
            y=y-y
            w=w-w
            break
        loss0=loss1
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, 10,False):
            # compute a stochastic gradient and loss
            grad, _ = compute_gradient_ridge(y_batch, tx_batch, w, lambda_)    
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
    loss = compute_loss_ridge(y, tx, initial_w, lambda_) 
    return w, loss

def ridge_sgd_hyperparamlambda(lambdas, gamma, nb_fold, nb_crossvalid, max_iters, x_train, y_train, w_initial):
    loss_valid = np.zeros([len(lambdas), nb_fold])
    loss_train = np.zeros([len(lambdas), nb_fold])
    
    nb_elem = math.floor(x_train.shape[0]/nb_fold)
    
    for i, lambda_ in enumerate(lambdas):
        for k in range(nb_crossvalid):
            x_valid_k = x_train[k*nb_elem:(k+1)*nb_elem][:]  
            y_valid_k = y_train[k*nb_elem:(k+1)*nb_elem]
            
            x_train_k = np.concatenate([x_train[0:k*nb_elem][:], x_train[(k+1)*nb_elem:][:]])
            y_train_k = np.concatenate([y_train[0:k*nb_elem],    y_train[(k+1)*nb_elem:]   ]) 
                                        
            w, loss_tr = ridge_SGD(y_train_k, x_train_k, w_initial, max_iters, gamma, lambda_)
            loss_train[i][k] = loss_tr
            loss_valid[i][k] = compute_loss_ridge(y_valid_k, x_valid_k, w, lambda_)
    ltrain = np.mean(loss_train, axis=1)
    lvalid = np.mean(loss_valid, axis=1)
    return lvalid, ltrain,w

def ridge_sgd_hyperparamgamma(lambda_, gammas, nb_fold, nb_crossvalid, max_iters, x_train, y_train, w_initial):
    loss_valid = np.zeros([len(gammas), nb_fold])
    loss_train = np.zeros([len(gammas), nb_fold])
    
    nb_elem = math.floor(x_train.shape[0]/nb_fold)
    
    for i, gamma in enumerate(gammas):
        for k in range(nb_crossvalid):
            x_valid_k = x_train[k*nb_elem:(k+1)*nb_elem][:]  
            y_valid_k = y_train[k*nb_elem:(k+1)*nb_elem]
            
            x_train_k = np.concatenate([x_train[0:k*nb_elem][:], x_train[(k+1)*nb_elem:][:]])
            y_train_k = np.concatenate([y_train[0:k*nb_elem],    y_train[(k+1)*nb_elem:]   ]) 
                                        
            w, loss_tr = ridge_SGD(y_train_k, x_train_k, w_initial, max_iters, gamma, lambda_)
            loss_train[i][k] = loss_tr
            loss_valid[i][k] = compute_loss_ridge(y_valid_k, x_valid_k, w, lambda_)
            
    ltrain = np.mean(loss_train, axis=1)
    lvalid = np.mean(loss_valid, axis=1)
    return lvalid, ltrain,w