# -*- coding: utf-8 -*-
import numpy as np
import math
from other import batch_iter
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission

# -----------------------------------------------------------
# --------------------Least squares -------------------------
# Common functions
def compute_gradient_least_squares(y, tx, w):
    """Compute the gradient of least square."""
    err = y - w.dot(tx.T)
    grad = -tx.T.dot(err)/len(err)
    return grad 

def compute_loss_ls(y, tx, w):
    """Calculate the loss of least squares."""
    e = y - tx.dot(w)
    #y_pred = predict_labels(w, tx)
    #accuracy = sum(y_pred == y)/len(y)
    return (1/2)*np.mean(e**2)
    #return accuracy

# Least squares
def least_squares(y, tx):
    """Compute the optimal w and the loss with least squares technique"""
    a = tx.T.dot(tx) 
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)                #p.7 du cours least squares
    loss = compute_loss_ls(y, tx, w)                 #p.3 du cours least squares
    return w, loss

# Gradient descent least squares
def least_squares_GD(y, tx, w_initial, max_iters, gamma):
    """Gradient descent algorithm with least squares."""
    w = w_initial
    for n_iter in range(max_iters):
        # compute gradient and error
        grad = compute_gradient_least_squares(y, tx, w)
        # gradient w by descent update
        w = w - (gamma * grad)  
    loss = compute_loss_ls(y, tx, w)
    return w, loss

def ls_gd_hyperparam(gammas, nb_fold,nb_crossvalid,max_iters, x_train, y_train,w_initial):
    loss_valid = np.zeros([len(gammas), nb_fold])
    loss_train = np.zeros([len(gammas), nb_fold])
    
    nb_elem = math.floor(x_train.shape[0]/nb_fold)
    
    for i, gamma in enumerate(gammas):
        for k in range(nb_crossvalid):
            x_valid_k = x_train[k*nb_elem:(k+1)*nb_elem][:]  
            y_valid_k = y_train[k*nb_elem:(k+1)*nb_elem]
            
            x_train_k = np.concatenate([x_train[0:k*nb_elem][:], x_train[(k+1)*nb_elem:][:]])
            y_train_k = np.concatenate([y_train[0:k*nb_elem],    y_train[(k+1)*nb_elem:]   ]) 
                                        
            w, loss_tr = least_squares_GD(y_train_k, x_train_k, w_initial, max_iters, gamma)
            loss_train[i][k] = loss_tr
            loss_valid[i][k] = compute_loss_ls(y_valid_k, x_valid_k, w)
            
    ltrain = np.mean(loss_train, axis=1)
    lvalid = np.mean(loss_valid, axis=1)        
    return lvalid, ltrain, w

# Stochactic gradient descent least squares
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic Gradient Descent algorithm with least squares."""
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

def ls_sgd_hyperparam(gammas, nb_fold, max_iters,x_train, y_train, w_initial):
    loss_valid = np.zeros([len(gammas), nb_fold])
    loss_train = np.zeros([len(gammas), nb_fold])
    
    nb_elem = math.floor(x_train.shape[0]/nb_fold)
    
    for i, gamma in enumerate(gammas):
        print(i)
        for k in range(nb_fold):
            x_valid_k = x_train[k*nb_elem:(k+1)*nb_elem][:]  
            y_valid_k = y_train[k*nb_elem:(k+1)*nb_elem]
            
            x_train_k = np.concatenate([x_train[0:k*nb_elem][:], x_train[(k+1)*nb_elem:][:]])
            y_train_k = np.concatenate([y_train[0:k*nb_elem],    y_train[(k+1)*nb_elem:]   ]) 
                                        
            w, loss_tr = least_squares_SGD(y_train_k, x_train_k, w_initial, max_iters, gamma)
            loss_train[i][k] = loss_tr
            loss_valid[i][k] = compute_loss_ls(y_valid_k, x_valid_k, w)
            
    return loss_valid, loss_train,w


# -----------------------------------------------------------
# -------------------- Ridge regression ---------------------

# Je suis un peu confuse. Est ce que ce qu'il veut c'est ça ou bien ce que j'ai fait aux fonctions ridge_GD et ridge_SGD ?? 
# ridge_GD et ridge_SGD sont juste en-dessous
def ridge_regression(y, tx, lambda_):
    """Compute Ridge regression."""
    lambd = 2 * tx.shape[0] * lambda_
    a = tx.T.dot(tx) + lambd * np.identity(tx.shape[1])
    b = tx.T.dot(y) 
    w = np.linalg.solve(a, b) 
    loss = compute_loss_ridge(y, tx, w, lambda_)   
    return w, loss

def ridge_GD(y, tx, initial_w, max_iters, gamma, lambda_):
    """Gradient descent algorithm with Ridge regression."""
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

def ridge_gd_hyperparam(lambdas,gammas, nb_fold,max_iters,x_train, y_train,w_initial):
    loss_train = np.zeros([len(gammas), len(lambdas), nb_fold])
    loss_valid = np.zeros([len(gammas), len(lambdas), nb_fold])
    
    nb_elem = math.floor(x_train.shape[0]/nb_fold)
    
    for i, gamma in enumerate(gammas):
        for j, lambda_ in enumerate(lambdas):
            for k in range(nb_fold):
                
                x_valid_k = x_train[k*nb_elem:(k+1)*nb_elem][:]  
                y_valid_k = y_train[k*nb_elem:(k+1)*nb_elem]

                x_train_k = np.concatenate([x_train[0:k*nb_elem][:], x_train[(k+1)*nb_elem:][:]])
                y_train_k = np.concatenate([y_train[0:k*nb_elem],    y_train[(k+1)*nb_elem:]   ]) 

                w, loss_gamma = ridge_SGD(y_train_k, x_train_k, w_initial, max_iters, gamma, lambda_)
                loss_train[i][j][k] = loss_gamma
                loss_valid[i][j][k] = compute_loss_ridge(y_valid_k, x_valid_k, w, lambda_)
                
    return loss_train, loss_valid, w

def ridge_SGD(y, tx, initial_w, max_iters, gamma, lambda_):
    """Stochastic Gradient Descent algorithm with least squares."""
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

def ridge_sgd_lambda(lambdas,gamma, nb_fold,max_iters, x_train, y_train, w_initial):
    loss_train = np.zeros([len(lambdas), nb_fold])
    loss_valid = np.zeros([len(lambdas), nb_fold])
    
    nb_elem = math.floor(x_train.shape[0]/nb_fold)
    
    
    for i, lambda_ in enumerate(lambdas):
        print(i)
        for k in range(nb_fold):
            x_valid_k = x_train[k*nb_elem:(k+1)*nb_elem][:]  
            y_valid_k = y_train[k*nb_elem:(k+1)*nb_elem]
            
            x_train_k = np.concatenate([x_train[0:k*nb_elem][:], x_train[(k+1)*nb_elem:][:]])
            y_train_k = np.concatenate([y_train[0:k*nb_elem],    y_train[(k+1)*nb_elem:]   ]) 
                                        
            w, loss_tr = ridge_SGD(y_train_k, x_train_k, w_initial, max_iters, gamma, lambda_)
            loss_train[i][k] = loss_tr
            loss_valid[i][k] = compute_loss_ridge(y_valid_k,x_valid_k,w,lambda_)

    return loss_train, loss_valid,w

# -----------------------------------------------------------
# --------------------Logistic regresion --------------------
def sigmoid(tx, w):
    """Compute sigmoid function"""
    z = np.array(np.exp(-tx.dot(w)))
    q = 1./(1+z)
    p = np.where(q < 0.99999999999, q, 0.99999999999)
    return p

def logistic_regression(y, tx, initial_w, max_iters, gamma): #SGD  (GD easy to implement from here)
    """Stochastic Gradient Descent algorithm with logistic regression."""
    w = initial_w
    batch_size=1
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient
            grad = compute_logreg_grad(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # compute a stochastic loss
            loss = compute_logreg_loss(y_batch, tx_batch, w)
    return w, loss

def compute_logreg_loss(y, tx, w): 
    """Compute loss of logistic regression"""
    sig = sigmoid(tx, w)
    loss = np.sum((-y * np.log(sig) - (1-y) * np.log(1-sig)), axis = -1)/len(y)
    return loss

def compute_logreg_grad(y, tx, w):
    """Compute gradient of logistic regression""" 
    sig = sigmoid(tx, w)
    err  = sig - y
    grad = tx.T.dot(err)
    return grad

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Stochastic Gradient Descent algorithm with REGULARIZED logistic regression."""
    """Required by project description"""
    w = initial_w
    batch_size=1
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient
            grad = compute_logreg_reg_grad(y_batch, tx_batch, w, lambda_)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # compute a stochastic loss
            loss = compute_logreg_reg_loss(y_batch, tx_batch, w, lambda_)
    return w, loss

def compute_logreg_reg_loss(y, tx, w, lambda_):
    """Compute lorr of regularized logistic regression"""
    reg = ( lambda_/(2*len(y)) ) * sum(w**2)
    loss = compute_logreg_loss(y, tx, w) + reg
    return loss

def compute_logreg_reg_grad(y, tx, w, lambda_):
    """Compute gradient of regularized logistic regression"""     
    grad = compute_logreg_grad(y, tx, w) 
    reg = (lambda_/len(y)) * w[1:]
    grad[1:] = grad[1:] + reg            
    return grad










                


def compute_loss_ridge(y, tx, w, lambda_):
    """Calculate the loss of ridge regression."""
    err = y - tx.dot(w)
    loss = (1/2) * np.mean(err**2) + lambda_ * (np.linalg.norm(w,2))**2   #TO CHECK p.3 ridge regression
    return loss


def compute_gradient_ridge(y, tx, w, lambda_):
    """Compute the gradient of ridge regression."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err) + (2 * lambda_* w)
    return grad, err



