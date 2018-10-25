from proj1_helpers import load_csv_data, create_csv_submission
from preprocessing import standardize_train, standardize_test
from run_fct import predict_labels, split_categories_run, remove_999col, add_bias
from implementations import ridge_GD
from other import replace_1



# Load training input
y_train, x_train, ids_train = load_csv_data('train.csv', sub_sample=False)
_,       x_test,  ids_test  = load_csv_data('test.csv',  sub_sample=False)


# Divide in categories, keep idx
cat_0_tr, cat_1_tr, cat_2_tr, cat_3_tr, idx_0_tr, idx_1_tr, idx_2_tr, idx_3_tr = split_categories_run(x_train)
cat_0_te, cat_1_te, cat_2_te, cat_3_te, idx_0_te, idx_1_te, idx_2_te, idx_3_te = split_categories_run(x_test)

# Adapt y_train to 0, 1 instead of 0, -1
y_train = replace_1(y_train)

# Select y_train for each cateory 
y_train_0 = y_train[idx_0_tr]
y_train_1 = y_train[idx_1_tr]
y_train_2 = y_train[idx_2_tr]
y_train_3 = y_train[idx_3_tr]

# Standardize data of each category individualy
cat_0_tr, mean_0, std_0 = standardize_train(cat_0_tr)
cat_0_te = standardize_test(cat_0_te)

cat_1_tr, mean_1, std_1 = standardize_train(cat_1_tr)
cat_1_te = standardize_test(cat_1_te)

cat_2_tr, mean_2, std_2 = standardize_train(cat_2_tr)
cat_2_te = standardize_test(cat_2_te)

cat_3_tr, mean_3, std_3 = standardize_train(cat_3_tr)
cat_3_te = standardize_test(cat_3_te)


#############################################################################
#############################################################################
#########         For each category, do some different stuff        #########
#########                        Change code                        #########
#############################################################################
#############################################################################

#############################################################################
#########                      For category 0                       #########
#############################################################################

## Preprocessing
# Remove columns with -999 values
cat_0_tr, cat_0_te = remove_999col(cat_0_tr, cat_0_te)
# Add bias 
cat_0_tr = add_bias(cat_0_tr)
cat_0_te = add_bias(cat_0_te)
# Build specific features
cat_0_tr = build_poly(cat_0_tr, degree = 1 )                        ######### ADAPT
cat_0_te = build_poly(cat_0_te, degree = 1 )                        ######### ADAPT
# Hyperparam definition ##################################################### CHANGE
gamma = 1
lambda_ = 1
# Computation
w_0, _ = ridge_SGD(y_train_0, cat_0_tr, initial_w, max_iters, gamma, lambda_)
#Prediction
y_0_te = predict_labels(w_0, cat_0_te)


#############################################################################
#########                      For category 1                       #########
#############################################################################

## Preprocessing
# Remove columns with -999 values
cat_1_tr, cat_1_te = remove_999col(cat_1_tr, cat_1_te)
# Add bias 
cat_1_tr = add_bias(cat_1_tr)
cat_1_te = add_bias(cat_1_te)
# Build specific features
cat_1_tr = build_poly(cat_1_tr, degree = 1 )                        ######### ADAPT
cat_1_te = build_poly(cat_1_te, degree = 1 )                        ######### ADAPT
# Hyperparam definition ##################################################### CHANGE
gamma = 1
lambda_ = 1
# Computation
w_1, _ = ridge_SGD(y_train_1, cat_1_tr, initial_w, max_iters, gamma, lambda_)
#Prediction
y_1_te = predict_labels(w_1, cat_1_te)


#############################################################################
#########                      For category 2                       #########
#############################################################################

## Preprocessing
# Remove columns with -999 values
cat_2_tr, cat_2_te = remove_999col(cat_2_tr, cat_2_te)
# Add bias 
cat_2_tr = add_bias(cat_2_tr)
cat_2_te = add_bias(cat_2_te)
# Build specific features
cat_2_tr = build_poly(cat_2_tr, degree = 1 )                        ######### ADAPT
cat_2_te = build_poly(cat_2_te, degree = 1 )                        ######### ADAPT
# Hyperparam definition ##################################################### CHANGE
gamma = 1
lambda_ = 1
# Computation
w_2, _ = ridge_SGD(y_train_2, cat_2_tr, initial_w, max_iters, gamma, lambda_)
#Prediction
y_2_te = predict_labels(w_2, cat_2_te)


#############################################################################
#########                      For category 3                       #########
#############################################################################

## Preprocessing
# Remove columns with -999 values
cat_3_tr, cat_3_te = remove_999col(cat_3_tr, cat_3_te)
# Add bias 
cat_3_tr = add_bias(cat_3_tr)
cat_3_te = add_bias(cat_3_te)
# Build specific features
cat_3_tr = build_poly(cat_3_tr, degree = 1 )                        ######### ADAPT
cat_3_te = build_poly(cat_3_te, degree = 1 )                        ######### ADAPT
# Hyperparam definition ##################################################### CHANGE
gamma = 1
lambda_ = 1
# Computation
w_3, _ = ridge_SGD(y_train_3, cat_3_tr, initial_w, max_iters, gamma, lambda_)
#Prediction
y_3_te = predict_labels(w_3, cat_2_te)



#############################################################################
#############################################################################
#############################################################################



# Reconstruct y in order
order_tab = np.concatenate((idx_0_te, idx_1_te, idx_2_te, idx_3_te))
order_idx = np.argsort(order_tab, axis=0)
y_unordered = np.concatenate((y_0_te, y_1_te, y_2_te, y_3_te))
y_pred = y_unordered[order_idx]


# Create submission
create_csv_submission(ids_test, y_pred, "submission")
