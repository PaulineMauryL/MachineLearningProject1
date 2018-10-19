import matplotlib.pyplot as plt


def log_plot_train_test(train_errors, test_errors, lambdas, title):
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

def plot_train_test(train_errors, test_errors, gammas):
    fig, ax = plt.subplots()
    ax.plot(gammas, train_errors, gammas, test_errors)

    ax.set(xlabel='Learning rate', ylabel='Loss',
           title='Learning rate on loss error (max_iter = 1000)')
    ax.grid()

    #fig.savefig("test.png")
    plt.show()