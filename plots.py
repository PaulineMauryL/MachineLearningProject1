import matplotlib.pyplot as plt


def log_plot_train_valid(train_errors, test_errors, lambdas, title):
    '''Plot in 2D the training and validation set error'''
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(lambdas, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("Learning rate")
    #plt.ylabel("RMSE")
    plt.ylabel("Loss")
    plt.title("Regularization parameter on loss error (max_iter = 1000)")
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    #plt.savefig("lest_squares")
    
#TODO plot 2D pour gammas / lambda

def plot_train_valid(train_errors, test_errors, gammas):
    fig, ax = plt.subplots()
    ax.plot(gammas, train_errors,'r-', gammas, test_errors,'bo')

    ax.set(xlabel='Learning rate', ylabel='Loss',
           title='Learning rate on loss error, train in red, valid in blue')
    ax.grid()
    
    fig.savefig("ls_200iter_gamma03_only999removed.png")
    plt.show()
    
def plot_train_valid_ridge(train_sgd_mean, valid_sgd_mean, lambdas):
    fig, ax = plt.subplots()
    ax.plot(lambdas, train_sgd_mean, label='Training set')
    ax.plot(lambdas, valid_sgd_mean, label = 'validing set')

    ax.set(xlabel='Lambdas', ylabel='Loss',
           title='Lambda on accuracy (max_iter = 200)')
    ax.grid()
    plt.xticks(lambdas)
    plt.ylim((0, 1))
    
    plt.legend()
    fig.savefig("ridge_lambda_20fold_200maxiter_only999removed.png")
    plt.show()