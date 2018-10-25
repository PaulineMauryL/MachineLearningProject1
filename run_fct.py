import numpy as np

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix
    Threshold at 0.5 because our predictions are between 0 and 1"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred




def split_categories_run(x_test):
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