from proj1_helpers import load_csv_data, create_csv_submission
from preprocessing import standardize_train, standardize_test


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix
    Threshold at 0.5 because our predictions are between 0 and 1"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred




def split_categories_test(x_test):
    jet_num = 22
    cat_0 = x_test[x_test[:, jet_num] == 0]
    cat_1 = x_test[x_test[:, jet_num] == 1]
    cat_2 = x_test[x_test[:, jet_num] == 2]
    cat_3 = x_test[x_test[:, jet_num] == 3]

    idx_0 = np.argwhere(x_test[:, jet_num] == 0)
    idx_1 = np.argwhere(x_test[:, jet_num] == 1)
    idx_2 = np.argwhere(x_test[:, jet_num] == 2)
    idx_3 = np.argwhere(x_test[:, jet_num] == 3)

    return cat_0, cat_1, cat_2, cat_3, idx_0, idx_1, idx_2, idx_3

# Load training input
y_train, x_train, ids_train = load_csv_data('train.csv', sub_sample=False)
_, x_test, ids_test = load_csv_data('test.csv', sub_sample=False)


# Divide in categories, keep idx
cat_0_tr, cat_1_tr, cat_2_tr, cat_3_tr, idx_0_tr, idx_1_tr, idx_2_tr, idx_3_tr = split_categories(x_train)
cat_0_te, cat_1_te, cat_2_te, cat_3_te, idx_0_te, idx_1_te, idx_2_te, idx_3_te = split_categories(x_test)

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



# Find weights 
w_0, _ = ridge_SGD(y_train_0, cat_0_tr, initial_w, max_iters, gamma, lambda_)
y_0 = predict_labels(w_0, cat_0_tr)

w_1, _ = ridge_SGD(y_train_1, cat_1_tr, initial_w, max_iters, gamma, lambda_)
y_1 = predict_labels(w_1, cat_1_tr)

w_2, _ = ridge_SGD(y_train_2, cat_2_tr, initial_w, max_iters, gamma, lambda_)
y_2 = predict_labels(w_2, cat_2_tr)

w_3, _ = ridge_SGD(y_train_3, cat_3_tr, initial_w, max_iters, gamma, lambda_)
y_3 = predict_labels(w_3, cat_2_tr)




def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids










#Il faut sortir une liste de ids and y_pred
def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})