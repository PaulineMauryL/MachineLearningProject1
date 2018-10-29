import numpy as np
import math
import matplotlib.pyplot as plt

def dataprocessing(cat_0_tri, cat_0_tei, degree, adddegree, inv, frac, sqroot, sqrootpos, cbroot, comb, comb3, trigo, expo, hyperb, combtrigo):
    '''Pre-processing of the data, load the useful features'''
    cat_0_tr, cat_0_te = remove_999col(cat_0_tri, cat_0_tei)
    
    to_add_cat_0_tr = build_data(cat_0_tr, degree, adddegree, inv, frac, sqroot, sqrootpos, cbroot, comb, comb3, trigo, expo, hyperb, combtrigo) 
    to_add_cat_0_te = build_data(cat_0_te, degree, adddegree, inv, frac, sqroot, sqrootpos, cbroot, comb, comb3, trigo, expo, hyperb, combtrigo)
    
    trx_0i = add_data(cat_0_tr, to_add_cat_0_tr)
    tex_0i = add_data(cat_0_te, to_add_cat_0_te)

    trx_0ii, mean0, std0 = standardize_train(trx_0i)
    trx_0 = add_bias(trx_0ii)

    tex_0 = add_bias(standardize_test(tex_0i, mean0, std0))
    
    return trx_0, tex_0

def split_categories(x_test):
	'''Split dataset in 4 different categories according to their jet number (column 22)'''
    jet_num = 22
    cat_0 = np.delete(x_test[x_test[:, jet_num] == 0],[22,29],axis=1)
    cat_1 = np.delete(x_test[x_test[:, jet_num] == 1],22,axis=1)
    cat_2 = np.delete(x_test[x_test[:, jet_num] == 2],22,axis=1)
    cat_3 = np.delete(x_test[x_test[:, jet_num] == 3],22,axis=1)

    idx_0 = np.argwhere(x_test[:, jet_num] == 0)
    idx_1 = np.argwhere(x_test[:, jet_num] == 1)
    idx_2 = np.argwhere(x_test[:, jet_num] == 2)
    idx_3 = np.argwhere(x_test[:, jet_num] == 3)
    

    return cat_0, cat_1, cat_2, cat_3, idx_0, idx_1, idx_2, idx_3

def standardize_train(x):
    ''' standardize training set
    '''
    centered_data = x - np.mean(x, axis=0)
    
    std_dev = np.std(x, axis=0)
    std_dev[std_dev == 0] = 1
    
    std_data = centered_data / std_dev
    
    return std_data, np.mean(x, axis=0), std_dev

def standardize_test(x, mean, std):
    ''' standardize test set with same values as training set
    '''
    return (x-mean)/std

def add_bias(x):
    '''add column vector of one for bias'''
    vect_one = np.ones([x.shape[0],1])
    return np.concatenate((vect_one, x), axis = 1)

def build_poly(tx, degree):
    """INPUT : matrix tx = [x1, x2, ..., xn]
       OUPUT : matrix tx = [x1^degree, x2^degree, ..., xn^degree] 
       """
    out = np.power(tx[:,0],degree)
    
    for k in np.arange(1, tx.shape[1]):
        out = np.c_[out, np.power(tx[:,k], degree)]
    
    return out

def build_fraction(tx):
    """INPUT : matrix tx = [x1, x2, ..., xn]
               with xi a column vector
               
       OUTPUT : [x1/x2 x1/x3 ... x1/xn, x2/x1, x2/x3, ..., x2/xn, ..., ..., xn/xn-1]
       """
    n = tx.shape[1] #n = nb_column
    out = tx[:,0]
    
    for i in np.arange(n):
        col = tx[:,i]
        sub_matrix = np.c_[ tx[:,0:i], tx[:,i+1:n]]
        to_add = build_inv( sub_matrix ) * col[:,np.newaxis] #Note that the order of matrix matters
        out = np.c_[out, to_add]
        
    end = out.shape[1]
    return out[:,1:end]

def build_data(tx, degree = False, adddegree = False, inv = False, frac = False, sqroot = False, sqrootpos = False, cbroot = False, comb = False, comb3 = False, trigo=False, expo = False, hyperb=False,combtrigo=False):
    """INPUT : matrix tx = [x1, x2, ... xn]
       OUTPUT : matrix = [1, x1, x1^2, ... x1^degree,   x2, x2^2, ..., x2^ degree, ...   , xn, xn^2, ... , x^degree]
       
       options :
       if sqrt = True, add the square root of each col : x1^0.5, x2^0.5, ... , xn^0.5
       if comb = True, add the linear combination of each col : (x1 * x2, x1 * x3, ... x1 *xn, .... , xn-1 * xn)
       """
    output = np.ones((tx.shape[0],1))
    #print("\n\n ON EST AU DEBUT DE BUILD_DATA")
    
    if degree:
        if adddegree:
            for i in range(1,degree,1):
                output = np.c_[output,build_poly(tx,i+1)]
        else:
            output = np.c_[output, build_poly(tx,degree)]

    if inv: #if el = 0, return 0 instead
        output = np.c_[output, build_inv(tx)]

    if frac:
        output = np.c_[output, build_fraction(tx)]
    
    if sqroot:
        output = np.c_[output, build_sqrt(tx)]
       
    if sqrootpos: #make sqrt( absolute_value ( tx ) )
        output = np.c_[output, np.sqrt( np.abs (tx) )]
  
    if cbroot:
        output = np.c_[output, np.cbrt(tx)]
      
    if comb:        
        output = np.c_[output, build_lin_com(tx)]
  
    if comb3:
        output = np.c_[output, build_all_deg_3(tx)]
        
    if trigo:
        output = np.c_[output, build_trigo(tx,0)]
     
    if expo:
        output = np.c_[output, np.exp(tx)]
        
    if hyperb:
        output = np.c_[output, build_hyperb(tx)]
        #print(output)
    #print("after hyperb, output = \n", output)
        
    if combtrigo:
        trigest= build_trigo(tx,1)
        step1= np.c_[tx,trigest]
        output = np.c_[output, build_lin_com(step1)]
    
    end = output.shape[1]
    #print("output before last part\n", output)
    output = output[:,1:end]
    #print("output after last part\n", output)
      
    return output

def build_lin_com(tx):
    """INPUT : matrix tx = [x1, x2, ... xn]
       OUTPUT : matrix output = [x1 x2, x1 x3, ... , x1 xn,     x2 x3, x2 x4, ... x2 xn, ...   , xn-1 xn]
       Note :  output has n(n-1)/2 columns"""
    output = tx[:,0]*tx[:,1]
    nb_col = tx.shape[1]
    end = int(nb_col*(nb_col-1)/2)+1
    
    for i in np.arange(nb_col-1):
        for j in np.arange(i+1,nb_col):
            output = np.c_[output, tx[:,i]*tx[:,j]]
    
    return output[:,1:end]

def build_sqrt(tx):
    """INPUT : matrix tx = [x1, x2, ..., xn]
       OUTPUT : matrix tx = [x1^0.5, x2^0.5, ..., xn^0.5]
       
       Note : if the value x is negativ, compute -|x|^0.5
       """
    out = np.where(tx < 0, -np.sqrt( np.abs(tx) ), np.sqrt(np.abs(tx)))
    return out

#def build_cbrt(tx):
#    """INPUT : matrix tx = [x1, x2, ..., xn]
#       OUTPUT : matrix tx = [x1^(1/3), x2^(1/3), ..., xn^(1/3)]
#   """
#    return np.cbrt(tx)
    
       
def add_data(tx, tx_to_add_to_tx):
    return np.c_[tx, tx_to_add_to_tx]

def build_lin_com3(tx):
    """ INPUT : tx = [x1 x2 x3 ... xn ]
        OUTPUT : out = [x1 x2 x3, x1 x2 x4, x1 x2 x5, ... x1 x2 xn, // x1 x3 x4, x1 x3 x5, ... , x1 x3 xn, // ... , ... , x1 xn-1 xn,
                        x2 x3 x4, x2 x3 x5, ... ... ... x2 xn-1 xn,
                        ...
                        xn-2 xn-1 xn]
                      
       Note : tx must at least have 3 columns !
    """ 
    n = tx.shape[1] #nb of column
    out = tx[:,0]*tx[:,1]*tx[:,2]
    
    for i in np.arange(n-2):
        #nb_el = i*(i-1)/2
        col = tx[:,i]
        to_add = build_lin_com( tx[:,i+1:n] ) * col[:, np.newaxis] #Note : the order for the multiplication matters !
        #print("\n\nto_add = \n", to_add)
        out = np.c_[out, to_add]
    
    end = out.shape[1]
    return out[:,1:end]

def build_lin_com_deg_old(tx,deg = 2):
    """ INPUT : tx = [x1 x2 x3 ... xn ]
        OUTPUT : out = [x1 x2^deg, x1 x3^deg, ..., x1 xn^deg,
                        x2 x3^deg, x2 x4^deg, ..., x2 xn^deg,
                        ...
                        xn-1 xn^deg ]
       C'est bien mais c'est pas ce qu'on veut |D
       Note : tx must has at least 2 columns !
       """ 
    n = tx.shape[1]
    out = np.ones((tx.shape[0],1))

    for i in np.arange(n-2):
        col = tx[:,i]
        pol_deg = build_poly(tx[:,i+1:n],deg)
        to_add = pol_deg*col[:, np.newaxis]
        out = np.c_[out, to_add]

    last = tx[:,tx.shape[1]-1]**deg * tx[:,tx.shape[1]-2]
    out = np.c_[out,last]

    end = out.shape[1]
    return out[:,1:end]

def build_lin_com_deg(tx,deg = 2):
    """ input : matrix tx = [x1 x2 ... xn] and degree desired
                with xi the column vector of tx
                
        output : every combination of each col to the power deg multiplied with one other column
                 i.e. output = [ x1^deg x2, x1^deg x3, ... , x1^deg xn, 
                                 x2^deg x1, x2^deg x3, ..., x2^deg xn,
                                 ...
                                 xn^deg x1, xn^deg x2, ..., xn^deg xn-1 ]
    """
    n = tx.shape[1]
    out = np.ones((tx.shape[0],1))
    
    for i in np.arange(n):
        core = np.c_[tx[:,0:i], tx[:,i+1:n]]
        col = tx[:,i]**deg
        to_add = core * col[:, np.newaxis]
        out = np.c_[out, to_add]
    
    end = out.shape[1]
    return out[:,1:end]

def build_all_deg_3(tx):
    """
    input : tx = [x1 x2 ... xn]
            with xi a column vector
            
    output : matrix of all combination of degree 3 : x1 x2 x3, x1^2 x3, ... 
             Note : it doesn't include cubic value (i.e. NO x1^3 !)
             
    Note : tx must at least have 3 columns
    """
    lin3 = build_lin_com3(tx)
    lin12 = build_lin_com_deg(tx,2)
    return np.c_[lin3,lin12]

def remove_999col(input_train,input_test):
	'''Remove columns that contain -999 in each categories'''
    idx = np.isin(input_train, -999.0)
    idx = np.any(idx,axis=0)
    ind = np.nonzero(idx)[0]
    
    x_train_no_999col = np.delete(input_train,ind,axis=1)
    x_test_no_999col = np.delete(input_test,ind,axis=1)
    
    return  x_train_no_999col,x_test_no_999col



def build_trigo(tx,num=0):
    if num:
        tx = np.c_[np.cos(tx),np.sin(tx),np.tan(tx)]
    else:
        tx = np.c_[np.cos(tx)]
    return tx

def build_hyperb(tx):
    out = np.c_[np.cosh(tx),np.sinh(tx),np.tanh(tx)]
    return out

def build_inv(tx):
    """ input tx = [x1 x2 ... xn]
    
        output = [1/x1 1/x2 ... 1/xn]
        
        Note : if el == 0, return 0 instead
        """
    nb_row = tx.shape[0]
    nb_col = tx.shape[1]

    out = np.zeros((nb_row,nb_col))

    for i in np.arange(nb_row):
        for j in np.arange(nb_col):
            if np.abs(tx[i,j]) > 1e-10:
                out[i,j] = np.reciprocal(tx[i,j])
                
    return out
    
    