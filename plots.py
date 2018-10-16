import matplotlib.pyplot as plt


def plot_train_test(train_errors, test_errors, gammas, title):
    plt.semilogx(gammas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(gammas, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("Learning rate")
    #plt.ylabel("RMSE")
    plt.ylabel("Loss")
    plt.title(title)
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    #plt.savefig("lest_squares")
    
#TODO plot 2D pour gammas / lambda