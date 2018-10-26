from proj1_helpers import load_csv_data
from implementations import ridge_regression, compute_loss_ridge
from other import accuracy
import numpy as np 
import math

def first_best(x_train, y_train,lambda_0):
    #print("First best")

    nb_fold = 3
    nb_elem = math.floor(x_train.shape[0]/nb_fold)
    nb_features = x_train.shape[1]
    lambda_ = lambda_0

    losses = []
    for feat in range(nb_features):     
        loss = []
        for k in range(nb_fold):
            x_valid_k = x_train[k*nb_elem:(k+1)*nb_elem][:,feat:feat+1]
            y_valid_k = y_train[k*nb_elem:(k+1)*nb_elem]
            
            x_train_k = np.concatenate([  x_train[0:k*nb_elem][:,feat:feat+1], x_train[(k+1)*nb_elem:][:,feat:feat+1] ])
            y_train_k = np.concatenate([  y_train[0:k*nb_elem],    y_train[(k+1)*nb_elem:]    ]) 
                                        
            w, _ = ridge_regression(y_train_k, x_train_k, lambda_)
            loss.append(compute_loss_ridge(y_valid_k, x_valid_k, w, lambda_))

        mean_loss = np.mean(loss)

        losses.append(mean_loss)

    min_feat = np.argmin(losses)
    min_loss = losses[min_feat]

    #print("Feature {} has the minimum loss of {}".format(min_feat, min_loss))

    return min_feat




def best_feature(x_train, y_train, nb_features_already, features_idx,lambda_0):
    #print("Best feature {} time".format(nb_features_already))

    nb_fold = 3
    nb_elem = math.floor(x_train.shape[0]/nb_fold)
    nb_features = x_train.shape[1]
    lambda_ = lambda_0
    x_train_already = x_train[:,features_idx]

    #print(x_train_already.shape)  #(250'000, 1)

    #accuracies =[]
    losses = []
    for feat in range(nb_features_already + 1, nb_features): 
        if(feat not in features_idx):
            x_train_cross = np.concatenate([x_train_already, x_train[:, feat:feat+1]], axis = 1)
            #print("Shape of x_train {}".format(x_train.shape))
            loss = []
            #acc = []
            for k in range(nb_fold):
                x_valid_k = x_train_cross[k*nb_elem:(k+1)*nb_elem][:]
                y_valid_k = y_train[k*nb_elem:(k+1)*nb_elem]
                
                x_train_k = np.concatenate([  x_train_cross[0:k*nb_elem][:], x_train_cross[(k+1)*nb_elem:][:] ])
                y_train_k = np.concatenate([  y_train[0:k*nb_elem],    y_train[(k+1)*nb_elem:]    ]) 
                                            
                w, _ = ridge_regression(y_train_k, x_train_k, lambda_)
                loss.append(compute_loss_ridge(y_valid_k, x_valid_k, w, lambda_))

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

    min_feat = np.argmin(losses)
    min_loss = losses[min_feat]
    #print("Feature {} has the minimum loss of {}".format(min_feat, min_loss))


    #using the accuracy would enable to break in the main loopwhen accuracy start decreasing but speed of algorithm is very affected. 
    #So keep loss and check afterwards where decrease
    return min_feat #max_feat, max_acc


def best_set_of_features(x_train, y_train,lambda_0):
    
    nb_features_already = 0
    first_min_feat = first_best(x_train, y_train,lambda_0)
    #print("first_mean_feat {}",format(first_min_feat))

    features_idx = []
    features_idx.append(first_min_feat)

    #x_train_new = x_train[:]  #################################
    #accuracy = []

    for i in range(20):

        nb_features_already += 1
        #x_train = x_train_new[:]
        #if(i==0):
        #feature, acc = best_feature(x_train, y_train, nb_features_already)
        feature = best_feature(x_train, y_train, nb_features_already, features_idx,lambda_0)
        #print("other feature {}",format(feature))
        ##if accuracy starts decreasing
        #if(i > 3 and acc < accuracy[-1] and acc < accuracy[-2]):
        #    break
        #accuracy.append(acc)
        features_idx.append(feature)
        #features_idx.append(feature - nb_features_already)
        
        #print("nb_features_already = {}".format(nb_features_already))

        #insérer la meileure feature au début de x_train et décaler toutes les autres vers la droite  (np.insert( dans x_train[0], x_train[feature] )  )
        '''
        for f in range(x_train.shape[1]):
            if(f==feature):
                x_train_new[0] = x_train[feature]
            elif(f<feature):
                x_train_new[f+1] = x_train[f]
            elif(f>feature):
                x_train_new[f] = x_train[f]
        '''
    return features_idx