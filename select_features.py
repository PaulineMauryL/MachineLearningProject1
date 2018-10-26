from proj1_helpers import load_csv_data
from implementations import ridge_regression, compute_loss_ridge
from implement import modeling
from other import accuracy
import numpy as np 
import math

def first_best(x_train, y_train,lambda_0):
    #print("First best")

    nb_fold = 3
    nb_elem = math.floor(x_train.shape[0]/nb_fold)

    losses = []
    for feat in range(x_train.shape[1]):     
        loss = []
        for k in range(nb_fold):
            x_valid_k = x_train[k*nb_elem:(k+1)*nb_elem][:,feat:feat+1]
            y_valid_k = y_train[k*nb_elem:(k+1)*nb_elem]
            
            x_train_k = np.concatenate([  x_train[0:k*nb_elem][:,feat:feat+1], x_train[(k+1)*nb_elem:][:,feat:feat+1] ])
            y_train_k = np.concatenate([  y_train[0:k*nb_elem],    y_train[(k+1)*nb_elem:]    ]) 
                                        
            w, _ = ridge_regression(y_train_k, x_train_k, lambda_0)
            loss.append(compute_loss_ridge(y_valid_k, x_valid_k, w, lambda_0))

        mean_loss = np.mean(loss)

        losses.append(mean_loss)

    min_feat = np.argmin(losses)
    min_loss = losses[min_feat]

    #print("Feature {} has the minimum loss of {}".format(min_feat, min_loss))

    return min_feat




def best_feature(x_train, y_train, nb_features_already, features_idx,lambda_0):

    nb_fold = 3
    nb_elem = math.floor(x_train.shape[0]/nb_fold)

    losses = []
    for feat in range(nb_features_already, x_train.shape[1]): 
        if(feat not in features_idx):
            x_train_cross = np.concatenate([x_train[:,features_idx], x_train[:, feat:feat+1]], axis = 1)
            #print("Shape of x_train {}".format(x_train.shape))
            loss = []
            #acc = []
            for k in range(nb_fold):
                x_valid_k = x_train_cross[k*nb_elem:(k+1)*nb_elem][:]
                y_valid_k = y_train[k*nb_elem:(k+1)*nb_elem]
                
                x_train_k = np.concatenate([  x_train_cross[0:k*nb_elem][:], x_train_cross[(k+1)*nb_elem:][:] ])
                y_train_k = np.concatenate([  y_train[0:k*nb_elem],    y_train[(k+1)*nb_elem:]    ]) 
                                            
                w, _ = ridge_regression(y_train_k, x_train_k, lambda_0)
                loss.append(compute_loss_ridge(y_valid_k, x_valid_k, w, lambda_0))

                #acc.append( accuracy(y_valid_k, x_valid_k, w) )

            #mean_acc = np.mean(acc)
            #print("mean accuracy is {}".format(mean_acc))
            #accuracies.append(mean_acc)
        
            mean_loss = np.mean(loss)
            #print("mean loss is {}".format(mean_loss))
            losses.append(mean_loss)
            
        elif(feat in features_idx):
            #print("feat in idx already")
            losses.append(1000)         # so high it will never be the min
    #max_feat = np.argmax(accuracies)
    #max_acc = accuracies[max_feat]
    #print("Feature {} has the max accuracy: {}".format(max_feat, max_acc))

    min_feat_rel = np.argmin(losses)
    print("min feature is {}".format(min_feat_rel))
    min_loss = losses[min_feat_rel]
    #print("Feature {} has the minimum loss of {}".format(min_feat, min_loss))
    min_feat = min_feat_rel + nb_features_already

    #using the accuracy would enable to break in the main loopwhen accuracy start decreasing but speed of algorithm is very affected. 
    #So keep loss and check afterwards where decrease
    return min_feat #max_feat, max_acc


def best_set_of_features(x_train, y_train):
    nb_fold = 10
    nb_crossvalid = 3
    
    nb_features_already = 0
    first_min_feat = first_best(x_train, y_train, lambda_0 = 0)
    #print("first_mean_feat {}",format(first_min_feat))

    features_idx = []
    features_idx.append(first_min_feat)
    
    lambdas = []
    _, lambda_best = modeling(x_train, y_train, num_intervals_lambda=10, nb_fold=60, nb_crossvalid=3, min_range=-2, max_range=0) 
    
    lambdas.append(lambda_best)
    
    accuracies = []
    
    for i in range(30):
        nb_features_already += 1

        feature = best_feature(x_train, y_train, nb_features_already, features_idx, lambda_best)
        
        x_train_modeling = np.concatenate([x_train[:,features_idx], x_train[:, feature:feature+1]], axis = 1)
        
        w_best, lambda_best = modeling(x_train_modeling, y_train, num_intervals_lambda=60, nb_fold=10, nb_crossvalid=3, min_range=-20, max_range=0) 
        lambdas.append(lambda_best)
        
        nb_elem = math.floor(x_train.shape[0]/nb_fold)
        acc = []
        for k in range(nb_crossvalid):
            x_valid_k = x_train[k*nb_elem:(k+1)*nb_elem][:]  
            y_valid_k = y_train[k*nb_elem:(k+1)*nb_elem]           
            x_train_k = np.concatenate([x_train[0:k*nb_elem][:], x_train[(k+1)*nb_elem:][:]])
            y_train_k = np.concatenate([y_train[0:k*nb_elem],    y_train[(k+1)*nb_elem:]   ]) 
            
            x_acc = np.concatenate([x_valid_k[:,features_idx], x_valid_k[:, feature:feature+1]], axis = 1)
            
            acc.append( accuracy(y_valid_k, x_acc, w_best) )
            
        print("Accuracy = {}".format(np.mean(acc)) )
        accuracies.append( np.mean(acc) )
        
        ##if accuracy starts decreasing
        if(i > 4 and accuracies[-1] < accuracies[-2] and accuracies[-1] < accuracies[-3]):
            print("Break, accuracy is decreasing since two lasts features (don't take last feature of list)")
            return features_idx, lambdas
        else:
            features_idx.append(feature)
        
    return features_idx, lambdas
