import numpy as np


def compute_loss_ls(y, tx, w):
    """Calculate the loss of least squares."""
    e = y - tx.dot(w)
    #print(e)
    return (1/2)*np.mean(e**2)


def compute_gradient_least_squares(y, tx, w):
    """Compute the gradient of least square."""
    err = y - w.dot(tx.T) #tx.dot(w)
    #print(err)
    #print(err.shape)
    grad = -tx.T.dot(err)/len(err)         #p.5 du cours least squares
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


def sigmoid(tx, w):
    """Compute sigmoid function"""
    z = np.array(np.exp(-tx.dot(w)))
    return 1./(1 + z)


def compute_logreg_loss(y, tx, w):  #np.log parce que math.log fonctionne pas.. J'ai toujours pas compris pourquoi.
    """Compute loss of logistic regression"""
    sig = sigmoid(tx, w)
    loss = np.sum((-y * np.log(sig) - (1-y) * np.log(1-sig)), axis = -1)/len(y)
    return loss

def compute_logreg_grad(y, tx, w):
    """Compute gradient of logistic regression""" 
    sig = sigmoid(tx, w)
    err  = sig - y
    grad = tx.T.dot(err)/len(y)
    return grad

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


