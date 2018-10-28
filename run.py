from proj1_helpers import load_csv_data, create_csv_submission, predict_labels
from preprocessing import dataprocessing, split_categories
from implementations import ridge_regression


# Load training input
y_train, x_train, ids_train = load_csv_data('train.csv', sub_sample=False)
_,       x_test,  ids_test  = load_csv_data('test.csv',  sub_sample=False)


# Divide in categories, keep idx
cat_0_tr, cat_1_tr, cat_2_tr, cat_3_tr, idx_0_tr, idx_1_tr, idx_2_tr, idx_3_tr = split_categories(x_train)
cat_0_te, cat_1_te, cat_2_te, cat_3_te, idx_0_te, idx_1_te, idx_2_te, idx_3_te = split_categories(x_test)

# Adapt y_train to 0, 1 instead of 0, -1
#y_train = replace_1(y_train)    pas besoin avec ridge

# Select y_train for each cateory 
y_train_0 = y_train[idx_0_tr]
y_train_1 = y_train[idx_1_tr]
y_train_2 = y_train[idx_2_tr]
y_train_3 = y_train[idx_3_tr]

# Pre-processing of each category
cat_0_tr, cat_0_te = dataprocessing(cat_0_tr, cat_0_te, degree = 5, adddegree = True, inv = True, frac = False, sqroot = True, sqrootpos = True, cbroot = True, comb = True, comb3 = True, trigo=True, expo = False, hyperb=False,combtrigo=False)
cat_1_tr, cat_1_te = dataprocessing(cat_1_tr, cat_1_te, degree = 5, adddegree = True, inv = True, frac = False, sqroot = True, sqrootpos = True, cbroot = True, comb = True, comb3 = True, trigo=True, expo = False, hyperb=False,combtrigo=False)
cat_2_tr, cat_2_te = dataprocessing(cat_2_tr, cat_2_te, degree = 5, adddegree = True, inv = True, frac = False, sqroot = True, sqrootpos = True, cbroot = True, comb = True, comb3 = True, trigo=True, expo = False, hyperb=False,combtrigo=False)
cat_3_tr, cat_3_te = dataprocessing(cat_3_tr, cat_3_te, degree = 5, adddegree = True, inv = True, frac = False, sqroot = True, sqrootpos = True, cbroot = True, comb = True, comb3 = True, trigo=True, expo = False, hyperb=False,combtrigo=False)

# Only keep useful columns of each category
feat_0 = []
feat_1 = []
feat_3 = []
feat_4 = []

cat_0_tr = cat_0_tr[:,feat_0]
cat_1_tr = cat_1_tr[:,feat_1]
cat_2_tr = cat_2_tr[:,feat_2]
cat_3_tr = cat_3_tr[:,feat_3]

cat_0_te = cat_0_te[:,feat_0]
cat_1_te = cat_1_te[:,feat_1]
cat_2_te = cat_2_te[:,feat_2]
cat_3_te = cat_3_te[:,feat_3]


# Define regularization parameter
lamb_0 = 
lamb_1 = 
lamb_2 = 
lamb_3 = 

# Find weights of each category
w_best_0, _ = ridge_regression(y_train_0, cat_0_tr, lamb_0)
w_best_1, _ = ridge_regression(y_train_1, cat_1_tr, lamb_1)
w_best_2, _ = ridge_regression(y_train_2, cat_2_tr, lamb_2)
w_best_3, _ = ridge_regression(y_train_3, cat_3_tr, lamb_3)


# Predict labels of each category
y_0_te = predict_labels(w_best_0, cat_0_te)
y_1_te = predict_labels(w_best_1, cat_1_te)
y_2_te = predict_labels(w_best_2, cat_2_te)
y_3_te = predict_labels(w_best_3, cat_3_te)


# Reconstruct y in order
order_tab = np.concatenate((idx_0_te, idx_1_te, idx_2_te, idx_3_te))
order_idx = np.argsort(order_tab, axis=0)
y_unordered = np.concatenate((y_0_te, y_1_te, y_2_te, y_3_te))
y_pred = y_unordered[order_idx]


# Create submission
create_csv_submission(ids_test, y_pred, "submission")
