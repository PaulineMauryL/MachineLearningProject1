import numpy as np

def compute_loss_ls(y, tx, w):
    """Calculate the loss.
    """
    e = y - tx.dot(w)
    #print(e)
    return (1/2)*np.mean(e**2)


def compute_gradient_least_squares(y, tx, w):
    """Compute the gradient of least square."""
    '''
    print("y is {}".format(y.shape))
    print("tx is {}".format(tx.shape))
    print("w is {}".format(w.shape))
    a = tx.dot(w)
    print("a is {}".format(a.shape))
    '''
    err = y - w.dot(tx.T) #tx.dot(w)
    #print(err)
    #print(err.shape)
    grad = -tx.T.dot(err)/len(err)         #p.5 du cours least squares
    return grad                  


def compute_loss_ridge(y, tx, w, lambda_):
    """Not required specifically""" 
    err = y - tx.dot(w)
    loss = (1/2) * np.mean(err**2) + lambda_ * (np.linalg.norm(w,2))**2   #TO CHECK p.3 ridge regression
    return loss


def compute_gradient_ridge(y, tx, w, lambda_):
    """Not required specifically""" 
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err) + (2 * lambda_* w)
    return grad, err


