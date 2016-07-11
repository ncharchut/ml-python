import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

############# USING NUMPY ################

def normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    normalized_X = (X-mu)/sigma
    return np.insert(normalized_X, 0, 1, axis=1)

def mean_square_error(Theta, X, y):
    m = len(y)
    # J = 1/(2*m) * sum((X*Theta - y)^2)
    return 1/(2.*m)*sum(np.square((np.subtract(X.dot(Theta),y))))

def gradient_descent(Theta, X, y, alpha=0.01, num_iterations=10000):
    m = len(y)
    J_cost = np.zeros((num_iterations,1))
    X = normalize(X)

    for i in xrange(num_iterations):
        # Theta = Theta - alpha/m * X'* (X * Theta - y)
        Theta = np.subtract(Theta, alpha/m*X.transpose().dot(np.subtract(X.dot(Theta),y)))
        J_cost[i] = mean_square_error(Theta, X, y)

    return Theta, J_cost, X

def sample_gradient_descent():
    xdata, ydata = [], []
    with open('data/ex1data1.txt','rU') as f:
        for row in f:
            x, y1 = row.split(',')
            xdata.append([float(x)])
            ydata.append([float(y1)])

    X = np.array(xdata)
    y = np.array(ydata)
    Theta = np.zeros((len(X[0])+1,1))
    Theta, J, X = gradient_descent(Theta, X, y)

    plt.title('Sample Gradient Descent',fontsize=28)
    plt.plot(X[:,1],y,'x',color='blue',label='Training Set')
    plt.plot(X[:,1], X.dot(Theta), color='red',label='Regressed Line')
    leg = plt.legend(fancybox=True,shadow=True,loc=0,framealpha=0.5)

    plt.show()
    
##########################################
